"""Utility to extract JSON from LLM completions."""

import re
import json
import ast


def extract_json(string, flag_longest=False) -> dict:
    """
    Extract JSON object from a string (typically an LLM completion).

    Args:
        string: The input string potentially containing JSON
        flag_longest: If True, use longest match pattern (currently same behavior)

    Returns:
        Extracted JSON as a dict, or None if extraction fails
    """
    if string is None:
        return None

    pattern = r"\{[\s\S]*?\}"

    matches = re.findall(pattern, string)

    if matches:
        for match in reversed(matches):  # Start with the last match
            try:
                extracted_json = json.loads(match)
                return extracted_json
            except json.JSONDecodeError:
                # try literal_eval if JSON loading fails
                try:
                    _json = ast.literal_eval(match.replace("'", '"'))
                    return _json
                except:
                    continue  # If this match fails, try the next one

        print("No valid JSON string found")
    else:
        print("No JSON string found")

    return None
