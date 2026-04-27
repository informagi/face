# Appendix

## Inference cost

One limitation of FACE is that it is more computationally expensive than direct LLM evaluators as it evaluates conversation particles with multiple optimized instructions and multiple samples.
Indeed, we could see FACE as a method of inference-time scaling paradigm [1].

For each evaluation aspect $\alpha$, FACE scores each conversation particle $p$ with the optimized instruction set $\mathbf{I}^{\alpha}$. In our experiments, the final instruction set has $|\mathbf{I}^{\alpha}| = 16$ instructions and the score distribution in $\mathcal{S}_{particle}(I^\alpha, p)$ is estimated with $n = 5$ samples. Therefore, the number of LLM scoring calls per aspect per particle is $|\mathbf{I}^{\alpha}| \times n = 16 \times 5 = 80$.

In CRSArena-Eval, the released FACE run files contain 6,274 particles over 2,220 scored assistant turns for the complete aspect files, corresponding to $|\mathbf{P}_x| = 2.83$ particles per assistant turn on average. This gives an average cost of $|\mathbf{P}_x| \times |\mathbf{I}^{\alpha}| \times n = 2.83 \times 16 \times 5 \approx 226$ LLM scoring calls per aspect per assistant turn.

This cost partly reflects the implementation. As discussed in the [FAQ](faq.md#what-is-equation-1-doing-does-face-run-the-same-prompt-multiple-times), $\mathcal{S}_{particle}(I^\alpha, p)$ was approximated by rerunning mulitiple prompts as the inference library we used, SGLang, did not expose the needed token probabilities at the time of the experiment; with such probabilities, the repeated calls can be replaced by a token-probability-weighted estimate.

## References

[1] "Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-Consistency Improves Chain of Thought Reasoning in Language Models. ICLR 2023.