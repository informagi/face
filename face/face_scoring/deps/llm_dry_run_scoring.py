"""
Dry run mock for FACE scoring LLM calls.

This module provides deterministic, mock responses for the FACE evaluation pipeline
when running in dry-run mode. It mimics the LLM's scoring behavior by returning
JSON responses containing the lowest possible score for the text's aspect, ensuring
evaluation components can be tested without actual model inference.

The main function `get_dry_run_scores` accepts a batch of prompts and the target aspect,
returning JSON strings that include the required score key and the minimum valid score.
"""

import json
from typing import List

try:
    import aspect_utils
except ImportError:
    aspect_utils = None

def get_dry_run_scores(prompts: List[str], aspect: str) -> List[str]:
    """
    Generate deterministic, batched mock responses for FACE scoring prompts.

    For each prompt, this function generates a JSON string where the score is set
    to the minimum possible value defined for the given aspect.

    Args:
        prompts: A list of prompt strings (instructions + dialogue context).
        aspect: The evaluation aspect (e.g., 'relevance', 'dialogue_overall').

    Returns:
        A list of JSON strings. Each string represents a valid LLM response
        containing the score key (e.g., "relevance_score": 0).
    """
    
    # Determine the minimum score for the aspect
    min_score = 0
    if aspect_utils and aspect in aspect_utils.pred_score_info:
        min_score = aspect_utils.pred_score_info[aspect]["min_score"]
    
    score_key = f"{aspect}_score"
    
    responses = []
    for _ in prompts:
        # Create a mock response with the lowest score
        mock_response = {
            score_key: min_score,
            "explanation": "Dry run deterministic output: lowest possible score."
        }
        responses.append(json.dumps(mock_response))
        
    return responses
