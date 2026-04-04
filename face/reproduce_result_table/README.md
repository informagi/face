# Results Calculation

This directory is for reproducing the main results (Table 2) from the paper.

## Directory Structure

```
run_files/
├── relevance.jsonl
├── interestingness.jsonl
├── understanding.jsonl
├── task_completion.jsonl
├── efficiency.jsonl
├── interest_arousal.jsonl
└── dialogue_overall.jsonl
```

The `run_files/` directory contains pre-computed scores for all **7 evaluation aspects**, separated into individual files:

| Aspect | Level | Description |
|--------|-------|-------------|
| `relevance` | Turn | Does the response make sense and meet the user's interests? |
| `interestingness` | Turn | Does the response make the user want to continue the conversation? |
| `understanding` | Dialogue | Does the assistant understand the user's request and try to fulfill it? |
| `task_completion` | Dialogue | Does the assistant make suggestions that the user finally accepts? |
| `efficiency` | Dialogue | Does the assistant suggest items matching user's interests within first three interactions? |
| `interest_arousal` | Dialogue | Does the assistant try to spark the user's interest in something new? |
| `dialogue_overall` | Dialogue | What is the overall impression of the assistant's performance? |

## Run File Format

Each aspect has its own [JSON Lines](https://jsonlines.org/) file (e.g., `relevance.jsonl`), where each line represents a single conversation:

```json
{"conv_id": "barcor_redial_...", "turns": {"1": {"particles": {"0": {"itr17": 1.4}}}}}
```

The structure of each line is:

```json
{
  "conv_id": "string",
  "turns": {
    "turn_ind": {
      "particles": {
        "particle_ind": {
          "run_id": score
        }
      }
    }
  }
}
```

### Components

- **conv_id**: Conversation identifier (e.g., `barcor_redial_...`); the same ID as original [CRSArena-Dial dataset](https://github.com/iai-group/crsarena-dial).
- **turn_ind**: Turn index (1-indexed string).
- **particle_ind**: Particle index (0-indexed string).
- **run_id**: Evaluation run identifier (e.g., `itr17`). Note that these are not consecutive integers because they represent the top-16 instructions selected from a larger pool of 96 candidate instructions during the optimization process; see Section 2.3 of the paper.
- **score**: The floating-point evaluation score.

## Conversation ID Format

The `conv_id` encodes information about the CRS system and training dataset:

```
{system}_{dataset}_{uuid}
```

**Systems** (from CRSArena-Dial):
- `barcor` - BARCOR CRS
- `unicrs` - UniCRS
- `kbrd` - KBRD
- `chatgpt` - ChatGPT-based CRS
- `crbcrs` - CRB-CRS (ReDial only)

**Datasets**:
- `redial` - ReDial dataset
- `opendialkg` - OpenDialKG dataset

Again, this is from the original [CRSArena-Dial dataset](https://github.com/iai-group/crsarena-dial).

## Conversation Particles

As described in Section 2.1 of the paper, FACE decomposes system responses into **conversation particles** - self-contained atomic information units consisting of:

1. **Dialogue Act**: The system's action (e.g., recommendation, preference elicitation)
2. **Mention**: The text span from the system's response
3. **Feedback**: The user's evaluative reply from the following turn

Each particle is independently evaluated using the optimized instructions, and the scores are aggregated to produce turn-level or dialogue-level evaluation scores.

## Reproducing Results

To reproduce the Table 2 correlations, run:

```bash
cd face/reproduce_result_table/
uv run python generate_result_table.py
```

This will:
1. Load all 16 run files per aspect
2. Aggregate particle → turn/dialogue scores
3. Average across the 16 instructions
4. Compute Pearson and Spearman correlations against human annotations
