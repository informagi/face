"""
Dry run mock for particle generation LLM calls.

This module provides deterministic, mock responses for the particle generation pipeline
when running in dry-run mode. It simulates the LLM's behavior by returning
syntactically valid JSON responses with "realistic" content based on simple heuristics,
bypassing actual API calls.

The main function `get_dry_run_particles` accepts a batch of prompts and returns
corresponding JSON strings that satisfy the ParticleValidator's requirements.
"""

import json
from typing import List, Dict

# Standard valid dialogue acts from nuggetize.py
VALID_DIALOGUE_ACTS = [
    "greeting",
    "preference elicitation",
    "recommendation",
    "goodbye",
    "others",
]

def get_dry_run_particles(prompts: List[str]) -> List[str]:
    """
    Generate deterministic, batched mock responses for particle generation prompts.

    For each prompt, this function generates a JSON string containing a list of particles.
    The content is determined deterministically based on the input prompt (e.g., using hashes
    or string lengths) to ensure consistency across runs.

    Args:
        prompts: A list of prompt strings sent to the LLM.

    Returns:
        A list of JSON strings, each simulating an LLM response with 'particle_generation_results'.
    """
    responses = []
    
    for i, prompt in enumerate(prompts):
        # Create a deterministic seed based on prompt length and index
        seed = len(prompt) + i
        
        # Pick a dialogue act based on the seed
        act_index = seed % len(VALID_DIALOGUE_ACTS)
        dialogue_act = VALID_DIALOGUE_ACTS[act_index]
        
        particle_text = f"dry run particle {i} for {dialogue_act}"
        
        mock_response = {
            "particle_generation_results": [
                {
                    "dialogue_act": dialogue_act,
                    "particle": particle_text,
                    "user_feedback": "N/A"
                }
            ]
        }
        
        responses.append(json.dumps(mock_response))
        
    return responses
