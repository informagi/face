# <div align="center"><img src="images/face_icon.png" alt="FACE icon" width="30" /> FACE: A Fine-grained Reference Free Evaluator for Conversational Recommender Systems<div>

<div align="center">
    <a href="https://arxiv.org/abs/2506.00314" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
    <a href="https://github.com/informagi/face">
        <img alt="License" src="https://img.shields.io/badge/License-MIT-blue?style=flat">
    </a>
</div>

This is the repo for our paper: **[FACE: A Fine-grained Reference Free Evaluator for Conversational Recommender Systems](https://arxiv.org/abs/2506.00314)**.

Specifically, the repository contains:  
- The [**`CRSArena-Eval dataset`**](dataset/) with human-annotated conversations and meta-evaluation scripts.
- The [**`CRSArena-Eval interface`**](interface/) for interactive meta-evaluation of your evaluator vs. baselines.
- The [**`FACE`**](face/) results for reproducing the reported numbers in the paper.

## What is CRSArena-Eval and FACE?

- **CRSArena-Eval** is a meta-evaluation dataset of human-annotated conversations between users and 9 Conversational Recommender Systems (CRSs), designed for evaluating CRS evaluators.
- **FACE** is a **Fine-grained, Aspect-based Conversation Evaluation** method that provides evaluation scores for diverse turn and dialogue level qualities of recommendation conversations.

## CRSArena-Eval Dataset Release (`dataset/`)

The directory [`dataset/`](dataset/) contains the **CRSArena-Eval dataset**.
This dataset is designed for **meta-evaluation** of CRS evaluators and is built on the [CRSArena-Dial dataset](https://github.com/iai-group/crsarena-dial).

- [**`crs_arena_eval.json`**](dataset/crs_arena_eval.json): The main dataset file containing 467 conversations with 4,473 utterances, annotated with both turn-level and dialogue-level quality scores by human evaluators.

### Evaluation Aspects

**Turn-level aspects**:
- **Relevance** (0-3): Does the assistant's response make sense and meet the user's interests?
- **Interestingness** (0-2): Does the response make the user want to continue the conversation?

**Dialogue-level aspects**:
- **Understanding** (0-2): Does the assistant understand the user's request and try to fulfill it?
- **Task Completion** (0-2): Does the assistant make recommendations that the user finally accepts?
- **Interest Arousal** (0-2): Does the assistant try to spark the user's interest in something new?
- **Efficiency** (0-1): Does the assistant suggest items matching the user's interests within the first three interactions?
- **Overall Impression** (0-4): What is the overall impression of the assistant's performance?

**Table: General statistics of the CRSArena-Eval dataset.**
| Statistic | Value |
| :--- | :--- |
| # Conversations | 467 |
| # Utterances | 4,473 |
| Avg. utterances per conversation | 9.58 |
| Avg. words per user utterance | 7.53 |
| Avg. words per system utterance | 15.18 |
| # Final labels (after aggregation) | 6,805 |

👉 For detailed dataset schema and structure, see [`dataset/README.md`](dataset/README.md).

### Evaluation

The [`dataset/run/`](dataset/run/) directory contains scripts and data for reproducing the evaluation results reported in the paper.

- [**`eval.py`**](dataset/run/eval.py): Evaluation script that computes Pearson and Spearman correlations between predictions and CRSArena-Eval human annotations.

- [**`face_run.json`**](dataset/run/face_run.json): FACE predictions for the CRSArena-Eval dataset in the standard run file format.


## CRSArena-Eval Interactive Meta-Evaluation Interface (`interface/`)

We provide an easy-to-use meta-evaluation interface to evaluate your evaluator against the CRSArena-Eval dataset.
Visit: https://informagus.nl/face/

![CRSArena-Eval demo](./dataset/run/demo.gif)

We also provide an python script to evaluate your evaluator on the CRSArena-Eval dataset.

👉 For detailed run file format and evaluation instructions, see [`dataset/run/README.md`](dataset/run/README.md).

## FACE Method (`face/`)

The [`face/`](face/) directory contains the implementation of the FACE evaluation method.


## Citation

```bibtex
@article{Joko:2025:FACE,
  title={FACE: A Fine-grained Reference Free Evaluator for Conversational Recommender Systems},
  author={Joko, Hideaki and Hasibi, Faegheh},
  journal={arXiv preprint arXiv:2506.00314},
  year={2025}
}
```

## Contact

If you have any questions, please contact Hideaki Joko (hideaki.joko@ru.nl)
