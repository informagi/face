# Prompt Examples

**This overview provides the prompts used in our method.**
The actual prompts used and optimized are available in each dir (`face_scoring`, `particle_generation`) respectively.

> **Note:** In all prompts, the term "nugget" refers to "conversation particle" as defined in the paper. All used and optimized prompts can be found in this repository.

---

## Conversation Particle Generation Prompt

> *Dialogue History:* `{dialogue_history}`  
> *Target Assistant Turn:* `{target_turn}`  
> *User's Response:* `{user_response}`
>
> **Your task is to extract conversation nuggets, which are minimal, atomic units of information or facts from the target assistant turn.**
>
> Each nugget consists of:
> - `"dialogue_act"`: one of the following labels: `"greeting"`, `"preference elicitation"`, `"recommendation"`, `"goodbye"`, or `"others"`.
> - `"nugget_mention"`: the atomic unit of information from the target assistant turn. [...]
> - `"user_feedback"`: the excerpt of user feedback against the given nugget. [...]
>
> The output must be a JSON list of nuggets. [...]
>
> Must think step by step:
> 1. Explain the dialogue history, the target assistant turn, and the user feedback.
> 2. How many conversation nuggets are found in the target assistant turn?
> 3. For each nugget, discuss the meaning of the user feedback.
> 4. Output in JSON format.

---

## Textual Gradient Prompt ∇

> **Examine the original instructions, predicted nugget score, and gold dialogue (or turn) score.**
>
> - Based on the gold dialogue (or turn) score, is the predicted nugget score reasonable?
> - Does original instructions describe how to use the nugget's information correctly?
> - Necessary to edit the original instructions?

---

## Instruction Rewriting Prompt δ

> **Propose new instructions of ~50 words based on the feedback.**
>
> - Note that the full dialogue can be changed, thus your new instructions must be general enough to handle different contexts.
> - Note that the task is "nugget" evaluation, not "turn" or "dialogue" evaluation; thus, the new instructions should focus on how to use the nugget.
> - Must provide "task description" and explicit "step-by-step instructions" for the nugget evaluation; in step-by-step instructions labeling each step as "Step 1," "Step 2," and so on.
> - Break down the evaluation into smaller steps and provide a checklist ("Does the nugget...?" or "Is this nugget...?") for each step.
> - [...]

---

## Initial Prompt (Before Optimization)

> **Task description:** Given the dialogue, evaluate the quality of the target nugget based on the `{evaluation_aspect}`.
>
> **Step-by-step instructions:**
>
> - **Step 1:** Read the dialogue history, target nugget, and user's response.
>   - What does the target nugget convey?
> - **Step 2:** Carefully read the grading criteria.
>   - What are the grading criteria?
> - **Step 3:** Evaluate the target nugget.
>   - Which grade should be assigned to the target nugget?

---

## FACE-Optimized Instructions

Here, we provide the optimized instruction examples for the dialogue-level overall impression aspect and the turn-level relevance aspect.
Please note that, in the actual process, FACE optimizes **multiple** instructions for each aspect.

### Optimized Instructions for Overall Impression Aspect (Dialogue-level)

> **Task description:** Evaluate the nugget based on its relevance, accuracy, and usefulness.
>
> **Step-by-step instructions:**
>
> - **Step 1:** Check if the nugget is relevant to the conversation.
>   - Does the nugget relate to the dialogue context?
>   - Is the nugget a direct response to the user's question or concern?
>   - Is the nugget related to the user's preferences or interests?
> - **Step 2:** Evaluate the nugget's accuracy.
>   - Is the information in the nugget accurate based on the dialogue?
>   - Does the nugget correctly represent the conversation?
> - **Step 3:** Assess the nugget's usefulness.
>   - Does the nugget provide a helpful or relevant suggestion?
>   - Does the nugget address the user's needs or concerns?
>   - Does the nugget facilitate a meaningful continuation of the conversation?

### Optimized Instructions for Relevance Aspect (Turn-level)

> **Task description:** Evaluate the quality of the target nugget based on its relevance to the user's request.
>
> **Step-by-step instructions:**
>
> - **Step 1:** Identify the user's request and the nugget's suggestion.
>   - Step 1.1: Does the nugget's suggestion directly address the user's request?
>   - Step 1.2: Is the nugget's genre or category aligned with the user's interest?
> - **Step 2:** Assess the nugget's relevance.
>   - Step 2.1: Does the nugget's information accurately address the user's need?
>   - Step 2.2: Is the nugget's suggestion consistent with the user's preferences or interests?
