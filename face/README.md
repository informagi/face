# FACE Method

This directory contains the implementation of the FACE (Fine-grained Aspect-based Conversation Evaluation) method. It provides tools for both **using FACE to evaluate conversations** and **reproducing the paper results**.

## Directory Structure

```
face/
├── particle_generation/     # Convert dialogue turns → atomic particles
│   ├── particle_generator.py   # Main generation script
│   └── deps/                    # Prompts and utilities
├── face_scoring/            # Score particles using optimized prompts
│   ├── face.py                  # Main scoring script
│   └── top_16_prompts/          # Pre-selected prompts per aspect
├── reproduce_result_table/  # Reproduce Table 2 correlations
│   ├── generate_result_table.py # Correlation computation script
│   └── run_files/               # Pre-computed FACE scores (JSONL)
│       └── particles/           # Exact particle run artifact for Table 2
├── results/                 # Output directory for new evaluations
└── prompt_overview.md       # Documentation of prompt structure
```

---

## Setup

1. **Install `uv`** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install Dependencies**:
   ```bash
   uv sync
   ```

3. **Configure LLM Backend**:  
   See [LLM Setup Guide](docs/llm_setup.md) for OpenRouter or local model (SGLang) configuration.

---

## 1. Particle Generation

Decomposes dialogue responses into atomic **conversation particles** -- self-contained information units consisting of:
- **Dialogue Act**: System action (e.g., recommendation, preference elicitation)
- **Mention**: Text span from the system's response
- **Feedback**: User's evaluative reply from the following turn

### Usage

From `face/`:

```bash
uv run particle_generation/particle_generator.py examples/example_conv.json \
    --turn-index 1 --speaker ASST --samples 10 --verbose
```

| Option | Description |
|--------|-------------|
| `utterance` | Text string OR path to conversation JSON file |
| `--turn-index` | Index of the turn to generate particles for (if using file) |
| `--speaker` | Speaker role (`USER` or `ASST`) |
| `--samples` | Number of LLM samples for plurality voting |
| `--dry-run` | Verify pipeline without API calls |

---

## 2. FACE Scoring

Evaluates particle-based dialogues using **16 optimized prompts** per aspect. Each particle is scored independently, then aggregated to turn/dialogue-level scores.

### Usage

From `face/`:

**Full Evaluation** (16 prompts × 5 samples):
```bash
uv run face_scoring/face.py --conversation examples/example_particles.json --aspect dialogue_overall
```

**Quick Check** (1 prompt × 1 sample):
```bash
uv run face_scoring/face.py --conversation examples/example_particles.json --aspect dialogue_overall \
    --samples 1 --max-prompts 1
```

| Option | Description |
|--------|-------------|
| `--conversation` | Path to particle-formatted conversation JSON |
| `--aspect` | Evaluation aspect (see below) |
| `--samples` | Samples per prompt-particle pair (default: 5) |
| `--max-prompts` | Limit number of prompts (default: 16) |
| `--dry-run` | Verify pipeline without API calls |

### Available Aspects

| Aspect | Level | Scale |
|--------|-------|-------|
| `relevance` | Turn | 0-3 |
| `interestingness` | Turn | 0-2 |
| `understanding` | Dialogue | 0-2 |
| `task_completion` | Dialogue | 0-2 |
| `efficiency` | Dialogue | 0-1 |
| `interest_arousal` | Dialogue | 0-2 |
| `dialogue_overall` | Dialogue | 0-4 |

---

## 3. Reproduce Results (Table 2)

Reproduces the main correlation results from the paper using pre-computed FACE scores.

### Usage

From `face/`:

```bash
uv run python reproduce_result_table/generate_result_table.py
```

This loads the 7 aspect-specific run files from `run_files/`, aggregates scores, and computes Pearson/Spearman correlations against CRSArena-Eval human annotations.

The exact particle text behind the released `particle_ind` references is available in `reproduce_result_table/run_files/particles/run_particles.json`.

### Run File Format

Each `run_files/{aspect}.jsonl` contains pre-computed scores:

```json
{"conv_id": "barcor_redial_...", "turns": {"1": {"particles": {"0": {"itr17": 1.4}}}}}
```

- **conv_id**: Conversation ID from CRSArena-Dial
- **turns → particles**: Nested structure mapping turn/particle indices to run scores
- **run_id** (e.g., `itr17`): Top-16 instruction IDs from the optimization process

For detailed format documentation, see [`reproduce_result_table/README.md`](reproduce_result_table/README.md).
