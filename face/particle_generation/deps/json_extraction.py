"""Utility to extract JSON from LLM completions."""

import json
import re
import ast


def extract_json(string, flag_longest=False) -> dict:
    """
    Extract JSON object from a string (typically an LLM completion).

    Args:
        string: The input string potentially containing JSON
        flag_longest: If True, use greedy matching; otherwise non-greedy

    Returns:
        Extracted JSON as a dict, or None if extraction fails
    """
    if string is None:
        return None

    if flag_longest:
        pattern = r"\{.*\}"
        matches = re.findall(pattern, string, re.DOTALL)
    else:
        pattern = r"\{.*?\}"
        matches = re.findall(pattern, string, re.DOTALL)

    if matches:
        json_string = matches[-1]

        try:
            extracted_json = json.loads(json_string)
            return extracted_json
        except json.JSONDecodeError:
            # try literal_eval
            try:
                _json = ast.literal_eval(json_string.replace("'", '"'))
                return _json
            except:
                pass
    return None
