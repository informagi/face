# FACE FAQ

This page collects short clarifications on FACE notation, particles, and scoring.

## What is a particle?

### What does a particle look like in practice?

From the released [particle artifact](../face/reproduce_result_table/run_files/particles/):

> What kind of vampire movies do you enjoy? Do you prefer classic vampire movies or more modern ones?

This turn is decomposed into:

```json
[
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
```

More examples are in the [particle artifact README](../face/reproduce_result_table/run_files/particles/README.md).

### What is the difference between an aspect and a particle?

- An **aspect** is an evaluation dimension such as `relevance` or `interest_arousal`. The released optimized prompts for each aspect are in the [top-16 prompt directory](../face/face_scoring/top_16_prompts/).
- A **particle** is the unit of evaluation. The released particle artifacts are in the [particle artifact directory](../face/reproduce_result_table/run_files/particles/).


### Is a particle the same thing as a nugget?

In FACE, we use the term **particle**. Earlier prompts and some older code used **nugget** for the same concept, however, we clarified and updated the term to **particle** because these units are decomposed evaluation units, not necessarily "gold nuggets" that we want to find in the system response.

## Notation and scoring

### What is the relationship between the particle sets in Sections 3.1 and 3.2?

The notation is:

$$
P_x =
\begin{cases}
P_r & \text{if the aspect is turn-level} \\
P_d & \text{if the aspect is dialogue-level}
\end{cases}
$$

Note that the $r$ in $P_r$ is unrelated to the $r$ in $r_p^a$. In $P_r$, $r$ refers to the system response; in $r_p^a$, it refers to the scoring LLM's response for particle $p$ under aspect $a$.

### What is the data type of $r_p^a$?

It is numeric. More specifically, in our experiments, it is an integer on a Likert-style scale, but the method should accommodate any numeric value.

### What is Equation (1) doing? Does FACE run the same prompt multiple times?

Equation (1) computes an expected score across multiple samples, and yes, in practice we run the same prompt multiple times.

Ideally, this should be computed with token probabilities, similar to [G-Eval](https://arxiv.org/abs/2303.16634). In our experiments, however, [SGLang](https://arxiv.org/abs/2312.07104) did not expose the token probabilities we needed at the time of our experiments, so we approximated it by rerunning the same prompt multiple times with the same input, which is the same approach taken by G-Eval.

### Why not just take the most frequent score?

The main reason is low variance. As noted in [G-Eval](https://arxiv.org/abs/2303.16634), "one digit usually dominates the distribution of the scores," and we observed the same general trend in our preliminary experiments.

The current implementation is computationally demanding, but that comes more from implementation constraints rather than from a methodological requirement. If the library exposes token probabilities, this can be replaced with a token-probability-weighted approach as in G-Eval.

### Why average across all particles instead of using a max over particles?

FACE currently averages across particles and reasoning paths (Section 3.2).
In our preliminary experiments, using a max did not show a performance improvement. One possible reason is that humans may penalize mistakes more than successes, although this may depend on the context (e.g., in recommendation, users may accept multiple imperfect recommendations if at least one is good).

Separately, in preliminary experiemnts, we also tested simple learned aggregators such as regression and decision trees, and did not see meaningful improvement there either, likely due to the limited number of conversations.

How to best aggregate across particles and reasoning paths is an open question, and could be an interesting direction for future work.

## Additional resources

### Is there a glossary or a longer notation guide?

Not yet but adding glossary is a great idea.
We may add a separate glossary or longer write-up later in this GitHub repository.


## Acknowledgment

This FAQ page was motivated by questions from [Prof. Laura Dietz](https://www.cs.unh.edu/~dietz/publications/index.html).
