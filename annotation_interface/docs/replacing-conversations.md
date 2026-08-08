# Replacing the conversations

Run these steps from the `annotation_interface/` directory.

1. Stop the app. It reads the conversations only once, during startup.
2. Prepare the new JSON file. The default `data/dialogue/CRSArena/sample_crs_arena_dial_with_votes.json` is only a two-dialogue sample.
3. If the hidden test changes, also replace:
   - `data/hidden_test/hidden_test_dial.json` (a one-item conversation list)
   - `data/hidden_test/gold_answer.json` (the corresponding gold annotation)
4. Validate the JSON, then restart the app:

   ```sh
   jq empty path/to/dialogues.json
   jq empty data/hidden_test/hidden_test_dial.json
   jq empty data/hidden_test/gold_answer.json
   uv run python app.py --dialogues path/to/dialogues.json
   ```

The main file must be a JSON list in the CRSArena-Dial format. Each item needs this structure:

```json
{
  "conversation ID": "system_name_user-123",
  "agent": {"id": "system_name", "type": "AGENT"},
  "user": {"id": "user-123", "type": "USER"},
  "metadata": {"sentiment": "neutral"},
  "conversation": [
    {"participant": "USER", "utterance": "Hello"},
    {"participant": "AGENT", "utterance": "Hi!"}
  ]
}
```

Keep every `conversation ID` unique and equal to `<agent.id>_<user.id>`. Turns must contain at least one `USER` and one `AGENT`, and their roles must alternate. Do not include the hidden-test conversation in the main file. `vote_result` and `utterance ID` are optional.

For a new annotation campaign, archive the existing `results/` directory before restarting. Otherwise, results whose IDs match the replacement data are counted as completed annotations, and old hidden-test results remain mixed with the new ones.
