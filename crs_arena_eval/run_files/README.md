# Run Files


This directory contains the run file of FACE, which is used to evaluate the performance against the CRSArena-Eval dataset.


## Run File Format

**There are two run files**, one for turn-level aspects and one for dialogue-level aspects.
These run files are JSON Lines files where each line is a JSON object representing the evaluation results for a single conversation turn or dialogue. Each object contains the following fields:

### Turn-level aspects

Aspects: `relevance` and `interestingness`

```python
{
  "conv_id": "string",  # Unique identifier for the conversation
  "turn_ind": int,      # Index of the turn being evaluated (0-based)
  "relevance": int,     # Relevance score (0-3)
  "interestingness": int # Interestingness score (0-2)
}
```

For turn-level aspect run file, the JSONL file should contain 2,235 lines in total.

### Dialogue-level aspects

Aspects: `understanding`, `task_completion`, `interest_arousal`, `efficiency`, and `overall_impression`:

```python
{
  "conv_id": "string",  # Unique identifier for the conversation
  "understanding": int, # Understanding score (0-2)
  "task_completion": int, # Task Completion score (0-2)
  "interest_arousal": int, # Interest Arousal score (0-2)
  "efficiency": int,    # Efficiency score (0-2)
  "overall_impression": int # Overall Impression score (0-2)
}
```

For dialogue-level aspect run file, the JSONL file should contain 467 lines in total.