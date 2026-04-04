"""Plurality vote handler for selecting the best nuggetization response."""

from typing import List, Dict
from collections import Counter


class PluralityVoteHandler:
    """Select the best particle generation response via plurality voting."""

    def __init__(self, responses: List[List[Dict]]):
        self.responses = responses

    def _sanitize_text(self, text: str) -> str:
        """Replace invalid Unicode characters with ''"""
        if not isinstance(text, str):
            return str(text)
        return text.encode('utf-8', errors='replace').decode('utf-8')

    def get_particle_counts(self) -> Dict[str, int]:
        """Count occurrences of unique particles"""
        particle_counts = Counter()
        for response in self.responses:
            for item in response:
                # Handle lagacy key if needed, or assume validator ensured 'particle' key
                content = item.get("particle", item.get("nugget"))
                if content:
                    particle_counts[content] += 1
        return dict(particle_counts)

    def get_combination_counts(self) -> Dict[tuple, int]:
        """Count occurrences of unique (action, particle, feedback) combinations"""
        combination_counts = Counter()
        for response in self.responses:
            for item in response:
                content = item.get("particle", item.get("nugget"))
                combination = (
                    item["dialogue_act"],
                    content,
                    item["user_feedback"]
                )
                combination_counts[combination] += 1
        return dict(combination_counts)

    def _calculate_response_total(self, response, total_combination_counts):
        """Calculate total votes for a response"""
        particle_votes = []
        for item in response:
            content = item.get("particle", item.get("nugget"))
            combination = (
                item["dialogue_act"],
                content,
                item["user_feedback"]
            )
            votes = total_combination_counts[combination]
            particle_votes.append(votes)
        return sum(particle_votes)

    def get_highest_voted_response(self):
        """Get the response with the most common particle count and highest votes.

        Returns:
            Tuple of (result_dict, total_votes) where result_dict contains
            'content' (the selected particles) and 'voting_stats'.
        """
        if not self.responses:
            return None

        # Step 1: Find the most common number of particles
        particle_counts = Counter(len(response) for response in self.responses)
        most_common_count = particle_counts.most_common(1)[0][0]

        # Filter responses to only those with the most common length
        candidate_responses = [
            response for response in self.responses
            if len(response) == most_common_count
        ]

        # Step 2: Calculate vote totals for candidates
        total_combination_counts = self.get_combination_counts()
        response_totals = [
            (response, self._calculate_response_total(response, total_combination_counts))
            for response in candidate_responses
        ]

        # Step 3: Select response with highest votes
        best_response, total_votes = max(response_totals, key=lambda x: x[1])

        result = {
            "content": best_response,
            "voting_stats": {
                "most_common_particle_count": most_common_count,
                "num_responses_with_this_count": particle_counts[most_common_count],
                "total_particle_votes": total_votes
            }
        }

        return result, total_votes
