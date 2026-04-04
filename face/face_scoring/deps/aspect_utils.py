"""Aspect definitions, score ranges, and grading descriptions for FACE evaluation."""

aspect_dict_nugget_based = {
    "relevance": "Does the target nugget make sense and meet the user's interests?",
    "interestingness": "Does the target nugget make the user want to continue the conversation?",
    "understanding": "As a whole, does the target nugget understand the user's request and try to fulfill it?",
    "task_completion": "Does the target nugget make recommendations that the user finally accepts?",
    "efficiency": "Does the target nugget make recommendations that fit the user's interests within the first three interactions?",
    "interest_arousal": "Does the target nugget try to spark the user's interest in something new?",
    "dialogue_overall": "What is the overall impression of the assistant's performance?"
}

level_to_aspects = {
    'turn_level': ['relevance', 'interestingness'],
    'dialogue_level': ['understanding', 'task_completion', 'efficiency', 'interest_arousal', 'dialogue_overall']
}

aspect_to_level = {aspect: level for level, aspects in level_to_aspects.items() for aspect in aspects}

# NOTE: 'non_applicable' will be ignored from the average score calculation.
# 'fall_back_for_all_non_applicable_turn' is used when all nuggets are non-applicable for the given turn.
pred_score_info = {
    'relevance': {'min_score': 0, 'max_score': 2, 'non_applicable': None, 'fall_back_for_all_non_applicable_turn': 0},
    'interestingness': {'min_score': 0, 'max_score': 2, 'non_applicable': None, 'fall_back_for_all_non_applicable_turn': 0},
    'understanding': {'min_score': 0, 'max_score': 2, 'non_applicable': None, 'fall_back_for_all_non_applicable_turn': 0},
    'task_completion': {'min_score': 0, 'max_score': 2, 'non_applicable': None, 'fall_back_for_all_non_applicable_turn': 0},
    'efficiency': {'min_score': 0, 'max_score': 1, 'non_applicable': None, 'fall_back_for_all_non_applicable_turn': 0},
    'interest_arousal': {'min_score': 0, 'max_score': 2, 'non_applicable': None, 'fall_back_for_all_non_applicable_turn': 0},
    'dialogue_overall': {'min_score': 0, 'max_score': 4, 'non_applicable': None, 'fall_back_for_all_non_applicable_turn': 0}
}

gold_score_info = {
    'relevance': {'min_score': 0, 'max_score': 2},
    'interestingness': {'min_score': 0, 'max_score': 2},
    'understanding': {'min_score': 0, 'max_score': 2},
    'task_completion': {'min_score': 0, 'max_score': 2},
    'efficiency': {'min_score': 0, 'max_score': 1},
    'interest_arousal': {'min_score': 0, 'max_score': 2},
    'dialogue_overall': {'min_score': 0, 'max_score': 4}
}

grading_description = {
    'relevance': {
        0: "No: The response is not relevant to the previous turn.",
        1: "Somewhat: The response is somewhat relevant to the previous turn.",
        2: "Yes: The response is relevant to the previous turn."
    },
    'interestingness': {
        0: "No: The response does not make chit-chat while presenting facts.",
        1: "Somewhat: The response somewhat makes chit-chat while presenting facts.",
        2: "Yes: The response makes chit-chat while presenting facts."
    },
    'understanding': {
        0: "No: The assistant does not understand the users request or fails to fulfill it.",
        1: "Somewhat: The assistant somewhat understands the users request or fulfills it partially.",
        2: "Yes: The assistant understands the users request and fulfills it."
    },
    'task_completion': {
        0: "No: The assistant does not make suggestions that the user accepts.",
        1: "Somewhat: The assistant makes suggestions, but the user only partially accepts them.",
        2: "Yes: The assistant makes suggestions that the user finally accepts."
    },
    'efficiency': {
        0: "No: The assistant cannot make good suggestions within the first three interactions.",
        1: "Yes: The assistant did make good suggestions within the first three interactions."
    },
    'interest_arousal': {
        0: "No: The assistant does not attempt to intrigue the user's interest.",
        1: "Somewhat: The assistant somewhat attempts to intrigue the user's interest.",
        2: "Yes: The assistant attempts to intrigue the user's interest."
    },
    'dialogue_overall': {
        0: "Very dissatisfied: Very dissatisfied with the assistant's performance",
        1: "Dissatisfied: Dissatisfied with the assistant's performance",
        2: "Neutral: Neutral with the assistant's performance",
        3: "Satisfied: Satisfied with the assistant's performance",
        4: "Very satisfied: Very satisfied with the assistant's performance"
    }
}

grading_description_nugget_based_v2 = {
    'relevance': {
        'null': "Not applicable: The nugget is not important for assessing relevance aspect of the assistant",
        0: "No: The nugget impacts the system's relevance quality negatively as it is not relevant to the previous turn.",
        1: "Somewhat: The nugget impacts the system's relevance quality somewhat negatively as it is somewhat relevant to the previous turn.",
        2: "Yes: The nugget impacts the system's relevance quality positively as it is relevant to the previous turn."
    },
    'interestingness': {
        'null': "Not applicable: The nugget is not important for assessing interestingness aspect of the assistant",
        0: "No: The nugget impacts the system's interestingness quality negatively as it does not engage in chit-chat while presenting facts.",
        1: "Somewhat: The nugget impacts the system's interestingness quality somewhat negatively as it somewhat engages in chit-chat while presenting facts.",
        2: "Yes: The nugget impacts the system's interestingness quality positively as it engages in chit-chat while presenting facts."
    },
    'understanding': {
        'null': "Not applicable: The nugget is not important for assessing understanding aspect of the assistant",
        0: "No: The nugget impacts the system's understanding quality negatively as it does not show understanding of the user's request or fails to address it.",
        1: "Somewhat: The nugget impacts the system's understanding quality somewhat negatively as it shows partial understanding of the user's request or addresses it partially.",
        2: "Yes: The nugget impacts the system's understanding quality positively as it shows clear understanding of the user's request and addresses it appropriately."
    },
    'task_completion': {
        'null': "Not applicable: The nugget is not important for assessing task completion aspect of the assistant",
        0: "No: The nugget impacts the system's task completion quality negatively as it does not contain suggestions that the user accepts.",
        1: "Somewhat: The nugget impacts the system's task completion quality somewhat negatively as it contains suggestions that the user only partially accepts.",
        2: "Yes: The nugget impacts the system's task completion quality positively as it contains suggestions that the user accepts."
    },
    'efficiency': {
        'null': "Not applicable: The nugget is not important for assessing efficiency aspect of the assistant",
        0: "No: The nugget impacts the system's efficiency quality negatively as it does not contribute to making good suggestions within the first three interactions.",
        1: "Yes: The nugget impacts the system's efficiency quality positively as it contributes to making good suggestions within the first three interactions."
    },
    'interest_arousal': {
        'null': "Not applicable: The nugget is not important for assessing interest arousal aspect of the assistant",
        0: "No: The nugget impacts the system's interest arousal quality negatively as it does not attempt to intrigue the user's interest.",
        1: "Somewhat: The nugget impacts the system's interest arousal quality somewhat negatively as it somewhat attempts to intrigue the user's interest.",
        2: "Yes: The nugget impacts the system's interest arousal quality positively as it attempts to intrigue the user's interest."
    },
    'dialogue_overall': {
        'null': "Not applicable: The nugget is not important for assessing dialogue overall aspect of the assistant",
        0: "Very dissatisfied: The nugget very negatively impacts the overall conversation.",
        1: "Dissatisfied: The nugget negatively impacts the overall conversation.",
        2: "Neutral: The nugget has a neutral impact on the overall conversation.",
        3: "Satisfied: The nugget positively impacts the overall conversation.",
        4: "Very satisfied: The nugget very positively impacts the overall conversation."
    }
}


def get_grading_description_string(aspect: str) -> str:
    """
    Returns a formatted string containing the grading descriptions for a given aspect.

    Args:
        aspect: The aspect to get descriptions for (e.g., 'relevance', 'interestingness', etc.)

    Returns:
        A formatted string with score descriptions, one per line
    """
    s = ""
    for score in range(gold_score_info[aspect]['min_score'], gold_score_info[aspect]['max_score'] + 1):
        s += f"- {score}: {grading_description[aspect][score]}\n"
    return s


def get_grading_description_string_nugget_based(aspect: str, include_null: bool = False) -> str:
    """
    Returns a formatted string containing the nugget-based grading descriptions for a given aspect.

    Args:
        aspect: The aspect to get descriptions for
        include_null: Whether to include the 'null' (not applicable) description

    Returns:
        A formatted string with score descriptions, one per line
    """
    grading_desc = grading_description_nugget_based_v2
    s = ""
    if include_null:
        s += f"- null: {grading_desc[aspect]['null']}\n"
    for score in range(gold_score_info[aspect]['min_score'], gold_score_info[aspect]['max_score'] + 1):
        s += f"- {score}: {grading_desc[aspect][score]}\n"
    return s
