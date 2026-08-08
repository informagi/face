import os
import re
import csv
from collections import defaultdict
from datetime import datetime
import random
import json

# global variables which stores active annotations
# TODO: This do not need to be a global variable since the instance of DialogueHandler is a global variable in app.py
ACTIVE_ANNOTATIONS = [] # list of {'conversation_id', 'start_time', 'annotator_id'}

class SanityCheck:
    def __init__(self, dialogues):
        self.dialogues = dialogues

    def check_alternating_roles(self):
        """Check if the dialogue has alternating roles, i.e. USER -> AGENT -> USER -> AGENT -> ..."""
        for dialogue in self.dialogues:
            conversation = dialogue['turns']
            for i in range(len(conversation) - 1):
                current_role = conversation[i].split(':')[0]
                next_role = conversation[i + 1].split(':')[0]
                if current_role == next_role:
                    return False
        return True

    def check_min_turns(self, min_turns=2):
        """Check if the dialogue has at least min_turns turns."""
        for dialogue in self.dialogues:
            if len(dialogue['turns']) < min_turns:
                return False
        return True

    def check_hidden_test_absence(self, hidden_test_id):
        """Check if the dialogue does not contain the hidden test conversation id."""
        for dialogue in self.dialogues:
            if dialogue['conversation_id'] == hidden_test_id:
                return False
        return True

    def run_all_checks(self, hidden_test_id):
        """Run all sanity checks."""
        checks = [
            ("Alternating roles", self.check_alternating_roles()),
            ("Minimum turns", self.check_min_turns()),
            ("Hidden test absence", self.check_hidden_test_absence(hidden_test_id))
        ]
        return all(result for _, result in checks), checks

class DialogueHandler:
    def __init__(
        self,
        logger,
        num_dialogues_per_worker,
        use_hidden_test,
        dialogues_path='data/dialogue/CRSArena/sample_crs_arena_dial_with_votes.json',
        hidden_test_path='data/hidden_test/hidden_test_dial.json',
    ):
        self.logger = logger
        self.num_dialogues_per_worker = num_dialogues_per_worker
        self.dialogues = self._read_dialogues(json_path=dialogues_path)
        self.conversation_ids = [dialogue['conversation_id'] for dialogue in self.dialogues]
        self.finished_annotation_counts = dict() # {conversation_id: finished_counts}
        self.assignment_history = self._initialize_assignment_history() # {'annotator_id': [conversation_id]} # list of conversation_id for each annotator to avoid assigning the same dialogue

        # refresh finished annotation counts
        self._refresh_finished_annotation_counts()

        # for hidden test if used
        self.use_hidden_test = use_hidden_test
        if self.use_hidden_test:
            self.hidden_test_dialogue = self._read_dialogues(json_path=hidden_test_path)[0] # only one dialogue
        
        self.sanity_checks()  # Move this line here

    def _read_dialogues(self, json_path) -> list:
        dialogues = []
        self.logger.info(f"Attempting to read JSON file from: {os.path.abspath(json_path)}")

        data = json.load(open(json_path, 'r', encoding='utf-8'))

        # check if all participants are either USER or AGENT
        assert {t['participant'] for d in data for t in d["conversation"]} == {"USER", "AGENT"}, f"Participants must be either USER or AGENT"
    
        for entry in data:
            dialogue = {
                "conversation_id": entry["conversation ID"],
                "user_id": entry["user"]["id"],
                "system_dataset": entry["agent"]["id"],
                "sentiment": entry["metadata"]["sentiment"],
                "vote": entry.get("vote_result", {}).get("result", ""),
                "feedback": entry.get("vote_result", {}).get("details", {}).get("feedback", ""),
                "turns": [f"{turn['participant']}: {turn['utterance']}" for turn in entry["conversation"]]
            }
            dialogues.append(dialogue)

        return dialogues

    def sanity_checks(self):
        sanity_check = SanityCheck(self.dialogues)
        all_passed, check_results = sanity_check.run_all_checks(self.get_hidden_test_conversation_id())
        
        for check_name, result in check_results:
            self.logger.info(f"Sanity check - {check_name}: {'Passed' if result else 'Failed'}")

        if all_passed:
            self.logger.info("Sanity check - All passed")
        else:
            raise ValueError("Dialogues did not pass sanity checks")
        

    def _read_finished_annotation_results(self, read_only_target_dialogues=True) -> list:
        """Read all annotation results from 'results/annotations' directory."""
        annotation_results = []
        annotation_dir = 'results/annotations'
        os.makedirs(annotation_dir, exist_ok=True)
        for filename in os.listdir(annotation_dir):
            if filename.endswith('.json'):
                match = re.search(r'(.+)_user-id:([^_]+)_annotator-id:([^_]+)', filename)
                if match:
                    system_dataset, user_id, annotator_id = match.groups()
                    conversation_id = f"{system_dataset}_{user_id}"
                    if read_only_target_dialogues and (conversation_id not in self.conversation_ids):
                        continue
                    annotation_results.append({
                        "conversation_id": f"{system_dataset}_{user_id}",
                        "annotator_id": annotator_id
                    })
        return annotation_results


    def _refresh_active_annotations(self, timeout_seconds=900) -> None:
        """Remove active annotation counts that have finished or unfinished after timeout_seconds"""
        global ACTIVE_ANNOTATIONS
        current_time = datetime.now()
        
        # Remove active annotation counts that have finished
        finished_annotations = self._read_finished_annotation_results()
        finished_annotations_tpl_set = {(annotation['conversation_id'], annotation['annotator_id']) for annotation in finished_annotations}
        ACTIVE_ANNOTATIONS = [
            annotation for annotation in ACTIVE_ANNOTATIONS
            if (annotation['conversation_id'], annotation['annotator_id']) not in finished_annotations_tpl_set
        ]
        
        # Remove active annotation counts that have been unfinished for more than timeout_seconds
        ACTIVE_ANNOTATIONS = [
            annotation for annotation in ACTIVE_ANNOTATIONS
            if (current_time - annotation['start_time']).total_seconds() <= timeout_seconds
        ]

    def _get_merged_annotation_counts(self,finished_annotation_counts, active_annotation_list, annotator_id) -> dict:
        """Merge active annotations and finished annotations, excluding previously assigned dialogues."""
        merged_annotation_counts = finished_annotation_counts.copy()
        for annotation in active_annotation_list:
            conversation_id = annotation['conversation_id']
            merged_annotation_counts[conversation_id] = merged_annotation_counts.get(conversation_id, 0) + 1
        
        # Exclude previously assigned dialogues for this annotator
        for conversation_id in self.assignment_history.get(annotator_id, []):
            merged_annotation_counts.pop(conversation_id, None)
        
        return merged_annotation_counts

    def _select_dialogue(self, active_annotations, annotator_id) -> dict:
        """Select a single dialogue for annotation."""
        # merge active annotations and finished annotations
        merged_annotation_counts = self._get_merged_annotation_counts(self.finished_annotation_counts, active_annotations, annotator_id)
        
        if not merged_annotation_counts:
            # If there are no annotations yet, select a random dialogue from all dialogues
            self.logger.info(f"No dialogues to annotate. Selecting a random dialogue from all dialogues")
            return random.choice(self.dialogues)
        
        # get the list of dialogues that have the least number of annotations
        min_count = min(merged_annotation_counts.values())
        self.logger.info(f"Minimum count: {min_count}")
        dialogues_with_min_count = [dialogue for dialogue, count in merged_annotation_counts.items() if count == min_count]

        # randomly select one dialogue key from the list
        selected_key = random.choice(dialogues_with_min_count)

        # find and return the corresponding dialogue object
        return next(dialogue for dialogue in self.dialogues if dialogue['conversation_id'] == selected_key)

    def _add_active_annotations(self, selected_dialogue, annotator_id) -> None:
        global ACTIVE_ANNOTATIONS
        start_time = datetime.now()
        conversation_id = selected_dialogue['conversation_id']
        ACTIVE_ANNOTATIONS.append({'conversation_id': conversation_id, 'start_time': start_time, 'annotator_id': annotator_id})

    def _refresh_finished_annotation_counts(self) -> None:
        annotation_results = self._read_finished_annotation_results()
        finished_annotation_counts = {conversation_id: 0 for conversation_id in self.conversation_ids}
        for result in annotation_results:
            finished_annotation_counts[result["conversation_id"]] += 1
        self.finished_annotation_counts = finished_annotation_counts # update the finished_annotation_counts

        
    def select_dialogue(self, annotator_id) -> dict:
        global ACTIVE_ANNOTATIONS
        self._refresh_active_annotations() # this must be first to avoid double counting
        self._refresh_finished_annotation_counts()

        # Select the first dialogue that hasn't reached the maximum annotation count
        selected_dialogue = self._select_dialogue(ACTIVE_ANNOTATIONS, annotator_id)

        # Log information about the selection
        conversation_id = selected_dialogue['conversation_id']
        finished_counts = self.finished_annotation_counts[conversation_id]
        active_counts = len([annotation for annotation in ACTIVE_ANNOTATIONS if annotation['conversation_id'] == conversation_id])
        self.logger.info(f"Selected dialogue {conversation_id} for annotation. Finished counts: {finished_counts}, Active counts: {active_counts}")

        # Add the selected dialogue to active annotations
        self._add_active_annotations(selected_dialogue, annotator_id)

        # Update assignment history
        self._update_assignment_history(annotator_id, conversation_id)

        # update progress
        self._progress_csv(self.finished_annotation_counts, ACTIVE_ANNOTATIONS)

        return selected_dialogue

    @staticmethod
    def _progress_csv(finished_annotation_counts, active_annotation_list) -> None:
        """Save annotation progress to csv file."""
        path_to_save = 'logs/progress.csv'
        os.makedirs('logs', exist_ok=True)
        with open(path_to_save, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["conversation_id", "finished_counts", "active_counts"])
            for conversation_id, finished_counts in finished_annotation_counts.items():
                active_counts = len([annotation for annotation in active_annotation_list if annotation['conversation_id'] == conversation_id])
                writer.writerow([conversation_id, finished_counts, active_counts])

    def _initialize_assignment_history(self) -> dict:
        finished_annotations = self._read_finished_annotation_results()
        assignment_history = defaultdict(list)
        
        for annotation in finished_annotations:
            assignment_history[annotation['annotator_id']].append(annotation['conversation_id'])
        
        return assignment_history

    def _update_assignment_history(self, annotator_id, conversation_id):
        """Update the assignment history for an annotator."""
        if annotator_id not in self.assignment_history:
            self.assignment_history[annotator_id] = []
        self.assignment_history[annotator_id].append(conversation_id)

    def get_hidden_test_conversation_id(self) -> str:
        return self.hidden_test_dialogue['conversation_id']

    def select_hidden_test_dialogue(self, annotator_id) -> dict:
        self._refresh_active_annotations()
        self._refresh_finished_annotation_counts()
        conversation_id = self.get_hidden_test_conversation_id()
        self._update_assignment_history(annotator_id, conversation_id)
        return self.hidden_test_dialogue
