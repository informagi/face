import argparse
import os
import json
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from src.dialogue_handler import DialogueHandler
from src.logger import Logger

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DIALOGUES_PATH = APP_DIR / 'data/dialogue/CRSArena/sample_crs_arena_dial_with_votes.json'
DEFAULT_HIDDEN_TEST_PATH = APP_DIR / 'data/hidden_test/hidden_test_dial.json'
DEFAULT_GOLD_ANSWER_PATH = APP_DIR / 'data/hidden_test/gold_answer.json'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the dialogue annotation interface.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        '--dialogues', type=Path, default=DEFAULT_DIALOGUES_PATH,
        help='JSON file containing the dialogues to annotate',
    )
    parser.add_argument(
        '--hidden-test', type=Path, default=DEFAULT_HIDDEN_TEST_PATH,
        help='JSON file containing the hidden-test dialogue',
    )
    parser.add_argument(
        '--gold-answer', type=Path, default=DEFAULT_GOLD_ANSWER_PATH,
        help='JSON file containing the hidden-test gold annotation',
    )
    args = parser.parse_args()

    def was_provided(option):
        return any(
            argument == option or argument.startswith(f'{option}=')
            for argument in sys.argv[1:]
        )

    hidden_test_provided = was_provided('--hidden-test')
    gold_answer_provided = was_provided('--gold-answer')
    if hidden_test_provided != gold_answer_provided:
        parser.error('--hidden-test and --gold-answer must be provided together')

    return args


args = parse_args()

# NUM_ANNOTATIONS_PER_DIALOGUE = 3
NUM_DIALOGUES_PER_WORKER = 20

# Hidden test
USE_HIDDEN_TEST = True
if USE_HIDDEN_TEST:
    hidden_test_dialogue_position = int(NUM_DIALOGUES_PER_WORKER/2)
    print(f'Hidden test is used. The dialogue is shown at {hidden_test_dialogue_position}-th dialogue.')



logger = Logger()
dialogue_handler = DialogueHandler(
    logger,
    NUM_DIALOGUES_PER_WORKER,
    USE_HIDDEN_TEST,
    dialogues_path=args.dialogues,
    hidden_test_path=args.hidden_test,
)

with args.gold_answer.open('r', encoding='utf-8') as gold_answer_file:
    gold_answer = json.load(gold_answer_file)

if USE_HIDDEN_TEST:
    gold_conversation_id = gold_answer['conversation']['conversation_id']
    if gold_conversation_id != dialogue_handler.get_hidden_test_conversation_id():
        raise ValueError('The gold answer and hidden-test dialogue must use the same conversation ID')

print(f'Dialogues: {args.dialogues.resolve()}')
print(f'Hidden test: {args.hidden_test.resolve()}')
print(f'Gold answer: {args.gold_answer.resolve()}')

app = Flask(__name__)
@app.route('/main-task.html')
def index():
    return render_template('index.html', num_dialogues_per_worker=NUM_DIALOGUES_PER_WORKER)

@app.route('/save_annotation', methods=['POST'])
def save_annotation():
    data = request.json
    json_data = data['data']
    
    # Parse the JSON data
    parsed_data = json.loads(json_data)
    
    # Extract components for the file name
    conversation_id = parsed_data['conversation']['conversation_id']
    system_dataset = parsed_data['conversation']['system_dataset']
    user_id = parsed_data['conversation']['user_id']
    # Raw results retain Prolific IDs; anonymize filenames and JSON content before public release.
    annotator_id = parsed_data['annotations']['annotator_prolific_id']
    timestamp = parsed_data['time']

    # Reconstruct the file name
    file_name = f"{system_dataset}_user-id:{user_id}_annotator-id:{annotator_id}_{timestamp}.json"
    # sample file name: barcor_opendialkg_user-id:bd9abd71-02e4-4bd3-aa79-748c3144f455_annotator-id:testmode_2024-09-18T23:02:38.115Z.json

    if USE_HIDDEN_TEST and dialogue_handler.get_hidden_test_conversation_id() == conversation_id:
        save_dir = 'results/annotations_hidden_test'
    else:
        save_dir = 'results/annotations'

    os.makedirs(save_dir, exist_ok=True)

    # Save the file
    file_path = os.path.join(save_dir, file_name)
    with open(file_path, 'w') as f:
        f.write(json_data)

    return jsonify({"message": "Annotation saved successfully"}), 200

@app.route('/instructions.html')
def quiz():
    return render_template('instructions.html')

@app.route('/get_next_dialogue', methods=['GET'])
def next_dialogue() -> dict:
    annotator_id = request.args.get('annotator_id')

    # If USE_HIDDEN_TEST and the current dialogue is the last one, show hidden test dialogue
    if USE_HIDDEN_TEST:
        current_dialogue_index = len(dialogue_handler.assignment_history.get(annotator_id, [])) + 1 # start from 1
        if current_dialogue_index == hidden_test_dialogue_position:
            dialogue = dialogue_handler.select_hidden_test_dialogue(annotator_id)
            return jsonify(dialogue)

    # Normal dialogue selection
    dialogue = dialogue_handler.select_dialogue(annotator_id)
    return jsonify(dialogue)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)
