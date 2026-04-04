"""Prompt template for single-utterance particle generation."""

prompt_template = """
Dialogue History:
{dialogue_history}

Target Assistant Turn:
{target_turn}

User's Response:
{user_response}

####

Your task is to extract information particles, which are minimal, atomic units of information or facts from the target assistant turn.

The information particles should be important for the evaluation of the assistant's performance.

Each particle consists of {{"dialogue_act": <str>, "particle": <str>, "user_feedback": <str>}}
- "dialogue_act": one of the following labels: "greeting," "preference elicitation," "recommendation," "goodbye," or "others."
- "particle": the atomic unit of information (1-12 words) from the target assistant turn. Keep the particle atomic; eliminate any unnecessary numbering or redundant information.
- "user_feedback": the excerpt of user feedback against the given particle. This should be an excerpt taken verbatim from the target assistant turn. If no user feedback is provided for the particle, use "NO_FEEDBACK_PROVIDED".

The output must be JSON list of particles, e.g.,
{{"particle_generation_results": [{{"dialogue_act": <str>, "particle": <str>, "user_feedback": <str>}}, ...]}}
If no particle is found, output the empty list `[]`.

Step 1: Explain the dialogue history, the target assistant turn, and the user feedback.
Step 2: How many information particles—atomic pieces of information—are found in the target assistant turn?
Step 3: For each particle, discuss the meaning of the user feedback.
Step 4: Output in JSON format.

Must think step by step:
"""
