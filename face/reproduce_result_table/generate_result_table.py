from __future__ import annotations

import json
import glob
from pathlib import Path
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr
import numpy as np
from typing import Dict, Tuple

# Aspect configuration
TURN_LEVEL_ASPECTS = ['relevance', 'interestingness']
DIALOGUE_LEVEL_ASPECTS = ['understanding', 'task_completion', 'efficiency', 'interest_arousal', 'dialogue_overall']
ALL_ASPECTS = TURN_LEVEL_ASPECTS + DIALOGUE_LEVEL_ASPECTS

# Paths relative to this script
SCRIPT_DIR = Path(__file__).parent
RUN_FILES_DIR = SCRIPT_DIR / "run_files"

def load_aspect_run_file(aspect: str) -> dict:
    """Load run file (JSONL) for a specific aspect."""
    file_path = RUN_FILES_DIR / f"{aspect}.jsonl"
    if not file_path.exists():
        raise FileNotFoundError(f"Run file not found for aspect '{aspect}' at {file_path}")
        
    print(f"Loading run file from {file_path}...")
    
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            conv_id = record.pop('conv_id')
            data[conv_id] = record
            
    return data


def load_and_aggregate_runs(aspect: str) -> dict:
    """
    Load predictions for an aspect from its run file and calculate averages.
    
    Returns dict mapping keys to averaged prediction scores.
    """
    data = load_aspect_run_file(aspect)
    is_turn_level = aspect in TURN_LEVEL_ASPECTS
    
    final_scores = {}
    
    for conv_id, conv_data in data.items():
        # Container for all scores relevant to this aspect for this conversation
        # If turn-level: map match key -> list of scores
        # If dialogue-level: simple list of scores
        if is_turn_level:
            conv_scores = defaultdict(list)
        else:
            conv_scores = []
            
        turns = conv_data.get("turns", {})
        for turn_ind, turn_data in turns.items():
            particles = turn_data.get("particles", {})
            for particle_ind, particle_data in particles.items():
                # particle_data is now a dict of run_id -> score
                scores = list(particle_data.values())
                
                if is_turn_level:
                    key = (conv_id, int(turn_ind))
                    conv_scores[key].extend(scores)
                else:
                    conv_scores.extend(scores)
        
        # Aggregate (average)
        if is_turn_level:
            for key, values in conv_scores.items():
                if values:
                    final_scores[key] = np.mean(values)
        else:
            if conv_scores:
                final_scores[conv_id] = np.mean(conv_scores)
                
    return final_scores


GOLD_DATA_PATH = SCRIPT_DIR.parent.parent / "dataset" / "crs_arena_eval.json"

def load_gold_scores(aspect: str) -> Tuple[Dict, Dict]:
    """
    Load gold annotations from the main dataset file.
    
    Returns:
        (redial_gold, opendialkg_gold) - dicts mapping keys to gold scores
    """
    is_turn_level = aspect in TURN_LEVEL_ASPECTS
    
    redial_gold = {}
    opendialkg_gold = {}
    
    if not GOLD_DATA_PATH.exists():
        raise FileNotFoundError(f"Gold data file not found at {GOLD_DATA_PATH}")

    with open(GOLD_DATA_PATH, 'r') as f:
        dialogues = json.load(f)
    
    for dialogue in dialogues:
        conv_id = dialogue['conv_id']
        
        # Determine dataset from conv_id
        if '_redial_' in conv_id:
            gold_dict = redial_gold
        elif '_opendialkg_' in conv_id:
            gold_dict = opendialkg_gold
        else:
            continue # specific handling or just skip if unexpected
            
        if is_turn_level:
            for turn in dialogue['dialogue']:
                # The gold file structure has 'aggregated' fields inside turns
                # specifically 'turn_level_aggregated'
                 if turn['role'] == 'ASST' and 'turn_level_aggregated' in turn:
                    aggregated = turn['turn_level_aggregated']
                    if aspect in aggregated:
                        key = (conv_id, turn['turn_ind'])
                        gold_dict[key] = aggregated[aspect]
        else:
            if 'dial_level_aggregated' in dialogue:
                aggregated = dialogue['dial_level_aggregated']
                if aspect in aggregated:
                    gold_dict[conv_id] = aggregated[aspect]
    
    return redial_gold, opendialkg_gold


def compute_correlations(pred_scores: dict, gold_scores: dict) -> dict:
    """
    Compute Pearson and Spearman correlations between predictions and gold.
    
    Returns dict with 'pearson' and 'spearman' keys.
    """
    # Get common keys
    common_keys = set(pred_scores.keys()) & set(gold_scores.keys())
    
    if len(common_keys) < 3:
        return {'pearson': float('nan'), 'spearman': float('nan'), 'n': len(common_keys)}
    
    preds = [pred_scores[k] for k in common_keys]
    golds = [gold_scores[k] for k in common_keys]
    
    pearson_r, _ = pearsonr(preds, golds)
    spearman_r, _ = spearmanr(preds, golds)
    
    return {
        'pearson': round(pearson_r, 3),
        'spearman': round(spearman_r, 3),
        'n': len(common_keys)
    }


def split_by_dataset(pred_scores: dict) -> Tuple[Dict, Dict]:
    """Split predictions by dataset (redial vs opendialkg) based on conv_id."""
    redial = {}
    opendialkg = {}
    
    for key, value in pred_scores.items():
        # Key is either (conv_id, turn_ind) or just conv_id
        conv_id = key[0] if isinstance(key, tuple) else key
        
        if '_redial_' in conv_id:
            redial[key] = value
        elif '_opendialkg_' in conv_id:
            opendialkg[key] = value
    
    return redial, opendialkg


def main():
    """Main function to reproduce Table 2 correlations."""
    print("=" * 80)
    print("FACE Table 2 Reproduction")
    print("=" * 80)
    print()
    
    results = {}
    
    for aspect in ALL_ASPECTS:
        print(f"Processing {aspect}...")
        
        # Load predictions
        pred_scores = load_and_aggregate_runs(aspect)
        pred_redial, pred_opendialkg = split_by_dataset(pred_scores)
        
        # Load gold scores
        gold_redial, gold_opendialkg = load_gold_scores(aspect)
        
        # Compute correlations
        results[aspect] = {
            'redial': compute_correlations(pred_redial, gold_redial),
            'opendialkg': compute_correlations(pred_opendialkg, gold_opendialkg)
        }
    
    print()
    print("=" * 80)
    print("Results: FACE Method Correlations with Human Annotations")
    print("=" * 80)
    print()
    
    # Print table header
    header = f"{'Aspect':<20} | {'ReDial r':<10} | {'ReDial ρ':<10} | {'OpenDialKG r':<12} | {'OpenDialKG ρ':<12}"
    print(header)
    print("-" * len(header))
    
    # Print results
    for aspect in ALL_ASPECTS:
        rd = results[aspect]['redial']
        kg = results[aspect]['opendialkg']
        print(f"{aspect:<20} | {rd['pearson']:<10.3f} | {rd['spearman']:<10.3f} | {kg['pearson']:<12.3f} | {kg['spearman']:<12.3f}")
    
    print()
    print("Legend: r = Pearson correlation, ρ = Spearman correlation")
    print()
    
    return results


if __name__ == "__main__":
    main()
