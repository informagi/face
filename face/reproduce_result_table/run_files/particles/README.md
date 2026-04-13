# FACE Particle Run File

This directory contains the particle run artifact used by the released FACE Table 2 reproduction files.

## File

- [`run_particles.json`](run_particles.json): one JSON Lines row per assistant utterance

## Particle Examples

```json
{
  "conv_id": "barcor_redial_03368a16-93bd-4b21-885d-b9a21e3498ba",
  "turn_ind": 1,
  "assistant_utterance": "Have you seen Blade Runner 2049 (2017)?",
  "particles": [
    {
      "particle_ind": 0,
      "dialogue_act": "recommendation",
      "particle": "Blade Runner 2049 (2017)",
      "user_feedback": "Could you provide me with a longer list"
    }
  ]
}
```

```json
{
  "conv_id": "chatgpt_redial_67db60a6-f6b3-4791-8d1c-edc5c9fbeb7a",
  "turn_ind": 3,
  "assistant_utterance": "What kind of vampire movies do you enjoy? Do you prefer classic vampire movies or more modern ones?",
  "particles": [
    {
      "particle_ind": 0,
      "dialogue_act": "preference elicitation",
      "particle": "What kind of vampire movies do you enjoy?",
      "user_feedback": "i like more modern ones"
    },
    {
      "particle_ind": 1,
      "dialogue_act": "preference elicitation",
      "particle": "Do you prefer classic vampire movies or more modern ones",
      "user_feedback": "i like more modern ones"
    }
  ]
}
```

## Statistics

| Metric | Count |
|--------|-------|
| Conversations | `467` |
| Assistant-turn rows | `2235` |
| Total particles | `6274` |
