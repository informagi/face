/*
 * Standalone survey data.
 *
 * This is a classic script rather than a fetched JSON resource so index.html
 * continues to work when opened directly with file:// and requires no server.
 *
 * Data model:
 * - papers: canonical paper metadata; titles and URLs are defined once
 * - columns: paper IDs shown in the matrix, in tie-break order
 * - aspects: the survey rows
 *
 * Aspect fields:
 * - number: stable tie-break order
 * - aspect: primary displayed name
 * - aliases: secondary names displayed in parentheses
 * - explanation: text, one or more paper citations, and an optional note
 * - paperMentions: paper ID to mark type ("check" or "related") and an
 *   optional paper-specific term shown in the cell tooltip
 */
window.FACE_SURVEY_DATA = {
  "schemaVersion": 1,
  "papers": [
    {
      "id": "sakai-23-swan",
      "citation": "Sakai 23",
      "columnLabel": "Sakai 23 · SWAN",
      "title": "SWAN: A Generic Framework for Auditing Textual Conversational Systems",
      "url": "https://arxiv.org/abs/2305.08290"
    },
    {
      "id": "gao22-cir",
      "citation": "Gao+22",
      "columnLabel": "Gao+22 · CIR",
      "title": "Neural Approaches to Conversational Information Retrieval",
      "url": "https://arxiv.org/abs/2201.05176"
    },
    {
      "id": "zamani22-cis",
      "citation": "Zamani+22",
      "columnLabel": "Zamani+22 · CIS",
      "title": "Conversational Information Seeking",
      "url": "https://arxiv.org/abs/2201.08808"
    },
    {
      "id": "mehri20-fed",
      "citation": "Mehri+20",
      "columnLabel": "Mehri+20 · FED",
      "title": "Unsupervised Evaluation of Interactive Dialog with DialoGPT",
      "url": "https://arxiv.org/abs/2006.12719"
    },
    {
      "id": "deriu20-eval-survey",
      "citation": "Deriu+20",
      "columnLabel": "Deriu+20 · Eval Survey",
      "title": "Survey on Evaluation Methods for Dialogue Systems",
      "url": "https://arxiv.org/abs/1905.04071"
    },
    {
      "id": "jannach-22-eval-crs",
      "citation": "Jannach 22",
      "columnLabel": "Jannach 22 · Eval CRS",
      "title": "Evaluating Conversational Recommender Systems: A Landscape of Research",
      "url": "https://arxiv.org/abs/2208.12061"
    },
    {
      "id": "hosking24-hfg",
      "citation": "Hosking+24",
      "columnLabel": "Hosking+24 · HFG",
      "title": "Human Feedback is not Gold Standard",
      "url": "https://arxiv.org/abs/2309.16349"
    },
    {
      "id": "xu23-critical-eval",
      "citation": "Xu+23",
      "columnLabel": "Xu+23 · Critical Eval",
      "title": "A Critical Evaluation of Evaluations for Long-form Question Answering",
      "url": "https://arxiv.org/abs/2305.18201"
    },
    {
      "id": "milano20-rs-ethics",
      "citation": "Milano+20",
      "columnLabel": "Milano+20 · RS Ethics",
      "title": "Recommender Systems and Their Ethical Challenges",
      "url": "https://doi.org/10.1007/s00146-020-00950-y"
    },
    {
      "id": "siro23-crs-satisfaction",
      "citation": "Siro+23",
      "columnLabel": "Siro+23 · CRS Satisfaction",
      "title": "Understanding and Predicting User Satisfaction with Conversational Recommender Systems",
      "url": "https://doi.org/10.1145/3624989"
    },
    {
      "id": "siro22-tds-satisfaction",
      "citation": "Siro+22",
      "columnLabel": "Siro+22 · TDS Satisfaction",
      "title": "Understanding User Satisfaction with Task-oriented Dialogue Systems",
      "url": "https://arxiv.org/abs/2204.12195"
    },
    {
      "id": "joko26-face",
      "citation": "Joko+26",
      "columnLabel": "Joko+26 · FACE",
      "title": "FACE: A Fine-Grained Reference-Free Evaluator for Conversational Information Access",
      "url": "https://arxiv.org/abs/2506.00314"
    },
    {
      "id": "joko24-laps",
      "citation": "Joko+24",
      "columnLabel": "Joko+24 · LAPS",
      "title": "Doing Personal LAPS: LLM-Augmented Dialogue Construction for Personalized Multi-Session Conversational Search",
      "url": "https://arxiv.org/abs/2405.03480"
    },
    {
      "id": "gao18-convai",
      "citation": "Gao+18",
      "columnLabel": "Gao+18 · ConvAI",
      "title": "Neural Approaches to Conversational AI",
      "url": "https://arxiv.org/abs/1809.08267"
    },
    {
      "id": "nvidia-24-nemotron-4",
      "citation": "NVIDIA 24",
      "columnLabel": "NVIDIA 24 · Nemotron-4",
      "title": "Nemotron-4 340B Technical Report",
      "url": "https://arxiv.org/abs/2406.11704"
    },
    {
      "id": "zhong22-unieval",
      "citation": "Zhong+22",
      "columnLabel": "Zhong+22 · UniEval",
      "title": "Towards a Unified Multi-Dimensional Evaluator for Text Generation",
      "url": "https://arxiv.org/abs/2210.07197"
    },
    {
      "id": "mehri20-usr",
      "citation": "Mehri+20",
      "columnLabel": "Mehri+20 · USR",
      "title": "USR: An Unsupervised and Reference Free Evaluation Metric for Dialog Generation",
      "url": "https://arxiv.org/abs/2005.00456"
    },
    {
      "id": "aliannejadi24-ikat",
      "citation": "Aliannejadi+24",
      "columnLabel": "Aliannejadi+24 · iKAT",
      "title": "TREC iKAT 2023: The Interactive Knowledge Assistance Track Overview",
      "url": "https://arxiv.org/abs/2401.01330"
    },
    {
      "id": "lin23-llm-eval",
      "citation": "Lin+23",
      "columnLabel": "Lin+23 · LLM-Eval",
      "title": "LLM-Eval: Unified Multi-Dimensional Automatic Evaluation for Open-Domain Conversations with Large Language Models",
      "url": "https://arxiv.org/abs/2305.13711"
    },
    {
      "id": "lee22-personachatgen",
      "citation": "Lee+22",
      "columnLabel": "Lee+22 · PersonaChatGen",
      "title": "PERSONACHATGEN: Generating Personalized Dialogues using GPT-3",
      "url": "https://aclanthology.org/2022.ccgpk-1.4/"
    },
    {
      "id": "li19-acute-eval",
      "citation": "Li+19",
      "columnLabel": "Li+19 · ACUTE-Eval",
      "title": "ACUTE-Eval: Improved Dialogue Evaluation with Optimized Questions and Multi-turn Comparisons",
      "url": "https://arxiv.org/abs/1909.03087"
    },
    {
      "id": "liu23-g-eval",
      "citation": "Liu+23",
      "columnLabel": "Liu+23 · G-Eval",
      "title": "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment",
      "url": "https://arxiv.org/abs/2303.16634"
    },
    {
      "id": "smith22-open-problem",
      "citation": "Smith+22",
      "columnLabel": "Smith+22 · Open Problem",
      "title": "Human Evaluation of Conversations is an Open Problem",
      "url": "https://arxiv.org/abs/2201.04723"
    }
  ],
  "columns": [
    "sakai-23-swan",
    "gao22-cir",
    "zamani22-cis",
    "mehri20-fed",
    "deriu20-eval-survey",
    "jannach-22-eval-crs",
    "hosking24-hfg",
    "xu23-critical-eval",
    "milano20-rs-ethics",
    "siro23-crs-satisfaction",
    "siro22-tds-satisfaction",
    "joko26-face",
    "joko24-laps",
    "gao18-convai",
    "nvidia-24-nemotron-4",
    "zhong22-unieval",
    "mehri20-usr",
    "aliannejadi24-ikat",
    "lin23-llm-eval",
    "lee22-personachatgen",
    "li19-acute-eval",
    "liu23-g-eval",
    "smith22-open-problem"
  ],
  "aspects": [
    {
      "number": 1,
      "aspect": "Relevance",
      "aliases": [],
      "explanation": {
        "text": "How well do the recommendations/responses provided by the system align with the user’s needs and preferences?",
        "citations": [
          {
            "paperId": "siro23-crs-satisfaction"
          },
          {
            "paperId": "zamani22-cis"
          }
        ]
      },
      "paperMentions": {
        "sakai-23-swan": {
          "mark": "check",
          "term": "Coherence"
        },
        "gao22-cir": {
          "mark": "check"
        },
        "zamani22-cis": {
          "mark": "check"
        },
        "mehri20-fed": {
          "mark": "check"
        },
        "deriu20-eval-survey": {
          "mark": "check"
        },
        "jannach-22-eval-crs": {
          "mark": "check",
          "term": "Accuracy"
        },
        "hosking24-hfg": {
          "mark": "check"
        },
        "xu23-critical-eval": {
          "mark": "check"
        },
        "siro23-crs-satisfaction": {
          "mark": "check"
        },
        "siro22-tds-satisfaction": {
          "mark": "check"
        },
        "joko26-face": {
          "mark": "check"
        },
        "joko24-laps": {
          "mark": "check"
        },
        "zhong22-unieval": {
          "mark": "check"
        },
        "aliannejadi24-ikat": {
          "mark": "check"
        },
        "lin23-llm-eval": {
          "mark": "check"
        }
      }
    },
    {
      "number": 2,
      "aspect": "Fluency",
      "aliases": [
        "Naturalness"
      ],
      "explanation": {
        "text": "How fluent and efficient is the system in communication in general?",
        "citations": [
          {
            "paperId": "zamani22-cis"
          }
        ]
      },
      "paperMentions": {
        "sakai-23-swan": {
          "mark": "check"
        },
        "gao22-cir": {
          "mark": "check"
        },
        "zamani22-cis": {
          "mark": "check"
        },
        "mehri20-fed": {
          "mark": "check"
        },
        "deriu20-eval-survey": {
          "mark": "check"
        },
        "jannach-22-eval-crs": {
          "mark": "check"
        },
        "hosking24-hfg": {
          "mark": "check"
        },
        "xu23-critical-eval": {
          "mark": "check"
        },
        "zhong22-unieval": {
          "mark": "check"
        },
        "mehri20-usr": {
          "mark": "check"
        },
        "aliannejadi24-ikat": {
          "mark": "check"
        },
        "lin23-llm-eval": {
          "mark": "check",
          "term": "Grammar"
        },
        "lee22-personachatgen": {
          "mark": "check"
        },
        "liu23-g-eval": {
          "mark": "check"
        }
      }
    },
    {
      "number": 3,
      "aspect": "Coherence",
      "aliases": [
        "Maintains Context"
      ],
      "explanation": {
        "text": "Does the response serve as a valid continuation of the previous conversation?",
        "citations": [
          {
            "paperId": "zhong22-unieval"
          }
        ]
      },
      "paperMentions": {
        "sakai-23-swan": {
          "mark": "check"
        },
        "gao22-cir": {
          "mark": "check"
        },
        "mehri20-fed": {
          "mark": "check",
          "term": "Coherent; Consistent"
        },
        "deriu20-eval-survey": {
          "mark": "check"
        },
        "jannach-22-eval-crs": {
          "mark": "check",
          "term": "Context compatibility"
        },
        "hosking24-hfg": {
          "mark": "check"
        },
        "joko24-laps": {
          "mark": "check"
        },
        "gao18-convai": {
          "mark": "check"
        },
        "nvidia-24-nemotron-4": {
          "mark": "check"
        },
        "zhong22-unieval": {
          "mark": "check"
        },
        "mehri20-usr": {
          "mark": "check"
        },
        "lin23-llm-eval": {
          "mark": "check",
          "term": "Appropriateness"
        }
      }
    },
    {
      "number": 4,
      "aspect": "Interestingness",
      "aliases": [
        "Engagingness"
      ],
      "explanation": {
        "text": "Does the system nugget/turn make the user want to continue the conversation?",
        "citations": [
          {
            "paperId": "sakai-23-swan"
          }
        ]
      },
      "paperMentions": {
        "sakai-23-swan": {
          "mark": "check"
        },
        "mehri20-fed": {
          "mark": "check"
        },
        "jannach-22-eval-crs": {
          "mark": "check",
          "term": "Novelty"
        },
        "siro23-crs-satisfaction": {
          "mark": "check"
        },
        "siro22-tds-satisfaction": {
          "mark": "check"
        },
        "joko26-face": {
          "mark": "check"
        },
        "zhong22-unieval": {
          "mark": "check"
        },
        "lee22-personachatgen": {
          "mark": "check"
        },
        "li19-acute-eval": {
          "mark": "check"
        },
        "liu23-g-eval": {
          "mark": "check"
        },
        "smith22-open-problem": {
          "mark": "check"
        }
      }
    },
    {
      "number": 5,
      "aspect": "Groundedness",
      "aliases": [
        "Consistency"
      ],
      "explanation": {
        "text": "Is the nugget based on some supporting evidence?",
        "citations": [
          {
            "paperId": "sakai-23-swan"
          }
        ]
      },
      "paperMentions": {
        "sakai-23-swan": {
          "mark": "check"
        },
        "zamani22-cis": {
          "mark": "check"
        },
        "deriu20-eval-survey": {
          "mark": "check"
        },
        "hosking24-hfg": {
          "mark": "check"
        },
        "zhong22-unieval": {
          "mark": "check"
        },
        "mehri20-usr": {
          "mark": "check",
          "term": "Uses Knowledge"
        },
        "aliannejadi24-ikat": {
          "mark": "check"
        },
        "liu23-g-eval": {
          "mark": "check"
        }
      }
    },
    {
      "number": 6,
      "aspect": "Correctness",
      "aliases": [
        "Factuality"
      ],
      "explanation": {
        "text": "Is the nugget factually correct?",
        "citations": [
          {
            "paperId": "sakai-23-swan"
          }
        ]
      },
      "paperMentions": {
        "sakai-23-swan": {
          "mark": "check"
        },
        "zamani22-cis": {
          "mark": "check"
        },
        "mehri20-fed": {
          "mark": "check"
        },
        "deriu20-eval-survey": {
          "mark": "check"
        },
        "hosking24-hfg": {
          "mark": "check"
        },
        "xu23-critical-eval": {
          "mark": "check"
        },
        "nvidia-24-nemotron-4": {
          "mark": "check"
        }
      }
    },
    {
      "number": 7,
      "aspect": "Personalisability",
      "aliases": [
        "Customisability",
        "Personalized relevance"
      ],
      "explanation": {
        "text": "Does the system adapt to different users and user groups?",
        "citations": [
          {
            "paperId": "sakai-23-swan"
          }
        ]
      },
      "paperMentions": {
        "sakai-23-swan": {
          "mark": "check"
        },
        "mehri20-fed": {
          "mark": "check",
          "term": "Flexible"
        },
        "jannach-22-eval-crs": {
          "mark": "check",
          "term": "Familiarity"
        },
        "joko24-laps": {
          "mark": "check"
        },
        "gao18-convai": {
          "mark": "check"
        },
        "aliannejadi24-ikat": {
          "mark": "check",
          "term": "PTKB statement ranking"
        },
        "lee22-personachatgen": {
          "mark": "check",
          "term": "Persona consistency"
        }
      }
    },
    {
      "number": 8,
      "aspect": "Effort and time",
      "aliases": [],
      "explanation": {
        "text": "How much effort and/or time was required to satisfy the information need?",
        "citations": [
          {
            "paperId": "zamani22-cis"
          }
        ]
      },
      "paperMentions": {
        "zamani22-cis": {
          "mark": "check"
        },
        "deriu20-eval-survey": {
          "mark": "check"
        },
        "jannach-22-eval-crs": {
          "mark": "check",
          "term": "Efciency of task support"
        },
        "siro23-crs-satisfaction": {
          "mark": "check",
          "term": "Efficiency"
        },
        "siro22-tds-satisfaction": {
          "mark": "check",
          "term": "Efficiency"
        },
        "joko26-face": {
          "mark": "check",
          "term": "Efficiency"
        },
        "gao18-convai": {
          "mark": "check"
        }
      }
    },
    {
      "number": 9,
      "aspect": "Interaction quality",
      "aliases": [],
      "explanation": {
        "text": "Approximation of user satisfaction.",
        "citations": [
          {
            "paperId": "deriu20-eval-survey"
          }
        ]
      },
      "paperMentions": {
        "gao22-cir": {
          "mark": "check"
        },
        "mehri20-fed": {
          "mark": "check",
          "term": "Overall impression"
        },
        "deriu20-eval-survey": {
          "mark": "check"
        },
        "jannach-22-eval-crs": {
          "mark": "check"
        },
        "siro23-crs-satisfaction": {
          "mark": "check",
          "term": "Overall impression"
        },
        "siro22-tds-satisfaction": {
          "mark": "check",
          "term": "Overall impression"
        },
        "joko26-face": {
          "mark": "check",
          "term": "Overall Impression"
        }
      }
    },
    {
      "number": 10,
      "aspect": "Understandable",
      "aliases": [
        "Sensible"
      ],
      "explanation": {
        "text": "Is the response understandable given the previous context?",
        "citations": [
          {
            "paperId": "mehri20-usr"
          }
        ]
      },
      "paperMentions": {
        "sakai-23-swan": {
          "mark": "check"
        },
        "mehri20-fed": {
          "mark": "check"
        },
        "deriu20-eval-survey": {
          "mark": "check"
        },
        "xu23-critical-eval": {
          "mark": "check"
        },
        "mehri20-usr": {
          "mark": "check"
        },
        "lin23-llm-eval": {
          "mark": "check",
          "term": "Content"
        }
      }
    },
    {
      "number": 11,
      "aspect": "Humanness",
      "aliases": [],
      "explanation": {
        "text": "The idea is to measure if the conversational dialogue system is capable of fooling a human into thinking that it is a human as well; i.e., Turing Test.",
        "citations": [
          {
            "paperId": "deriu20-eval-survey"
          }
        ]
      },
      "paperMentions": {
        "deriu20-eval-survey": {
          "mark": "check"
        },
        "jannach-22-eval-crs": {
          "mark": "check"
        },
        "gao18-convai": {
          "mark": "check"
        },
        "lee22-personachatgen": {
          "mark": "related"
        },
        "li19-acute-eval": {
          "mark": "check"
        },
        "smith22-open-problem": {
          "mark": "check"
        }
      }
    },
    {
      "number": 12,
      "aspect": "Safety",
      "aliases": [],
      "explanation": {
        "text": "No threats, no insults, no hate or harassment, etc.",
        "citations": [
          {
            "paperId": "sakai-23-swan"
          }
        ]
      },
      "paperMentions": {
        "sakai-23-swan": {
          "mark": "check"
        },
        "zamani22-cis": {
          "mark": "related"
        },
        "hosking24-hfg": {
          "mark": "check",
          "term": "Harmful"
        },
        "milano20-rs-ethics": {
          "mark": "check"
        },
        "nvidia-24-nemotron-4": {
          "mark": "check"
        }
      }
    },
    {
      "number": 13,
      "aspect": "Task-success",
      "aliases": [],
      "explanation": {
        "text": "How well the dialogue system fulfills the information requirements dictated by the user’s goals.  (Similar to #29)",
        "citations": [
          {
            "paperId": "deriu20-eval-survey"
          }
        ]
      },
      "paperMentions": {
        "deriu20-eval-survey": {
          "mark": "check"
        },
        "siro23-crs-satisfaction": {
          "mark": "check",
          "term": "Task completion"
        },
        "siro22-tds-satisfaction": {
          "mark": "check",
          "term": "Task completion"
        },
        "joko26-face": {
          "mark": "check",
          "term": "Task Completion"
        },
        "gao18-convai": {
          "mark": "check"
        }
      }
    },
    {
      "number": 14,
      "aspect": "Explainability",
      "aliases": [
        "Rationale"
      ],
      "explanation": {
        "text": "Can the user see how the system came up with the nugget?",
        "citations": [
          {
            "paperId": "sakai-23-swan"
          }
        ]
      },
      "paperMentions": {
        "sakai-23-swan": {
          "mark": "check"
        },
        "zamani22-cis": {
          "mark": "check"
        },
        "milano20-rs-ethics": {
          "mark": "check"
        },
        "joko24-laps": {
          "mark": "check"
        }
      }
    },
    {
      "number": 15,
      "aspect": "Conciseness",
      "aliases": [],
      "explanation": {
        "text": "Is the system turn minimal in length?",
        "citations": [
          {
            "paperId": "sakai-23-swan"
          }
        ]
      },
      "paperMentions": {
        "sakai-23-swan": {
          "mark": "check"
        },
        "deriu20-eval-survey": {
          "mark": "check"
        },
        "xu23-critical-eval": {
          "mark": "check"
        },
        "nvidia-24-nemotron-4": {
          "mark": "check",
          "term": "Verbosity"
        }
      }
    },
    {
      "number": 16,
      "aspect": "Fair exposure",
      "aliases": [
        "Bias"
      ],
      "explanation": {
        "text": "Does the system mention different groups fairly across its turns?",
        "citations": [
          {
            "paperId": "sakai-23-swan"
          }
        ]
      },
      "paperMentions": {
        "sakai-23-swan": {
          "mark": "check"
        },
        "gao22-cir": {
          "mark": "check"
        },
        "zamani22-cis": {
          "mark": "check"
        },
        "milano20-rs-ethics": {
          "mark": "check"
        }
      }
    },
    {
      "number": 17,
      "aspect": "User satisfaction",
      "aliases": [
        "frustration"
      ],
      "explanation": {
        "text": "Is the user satisfied with the outcome of the conversation? Was the user frustrated in the process?",
        "citations": [
          {
            "paperId": "zamani22-cis"
          }
        ]
      },
      "paperMentions": {
        "gao22-cir": {
          "mark": "check"
        },
        "zamani22-cis": {
          "mark": "check"
        },
        "deriu20-eval-survey": {
          "mark": "check"
        },
        "jannach-22-eval-crs": {
          "mark": "check",
          "term": "Efectiveness of task support"
        }
      }
    },
    {
      "number": 18,
      "aspect": "Response diversity",
      "aliases": [],
      "explanation": {
        "text": "The capability of a system to create diversifed, yet relevant recommendation lists (or respnoses).",
        "citations": [
          {
            "paperId": "jannach-22-eval-crs"
          }
        ]
      },
      "paperMentions": {
        "mehri20-fed": {
          "mark": "check",
          "term": "Diverse"
        },
        "jannach-22-eval-crs": {
          "mark": "check"
        },
        "joko24-laps": {
          "mark": "related"
        },
        "gao18-convai": {
          "mark": "check"
        }
      }
    },
    {
      "number": 19,
      "aspect": "Serendipity",
      "aliases": [],
      "explanation": {
        "text": "The ability of a system to recommend surprising, yet relevant content",
        "citations": [
          {
            "paperId": "jannach-22-eval-crs"
          }
        ]
      },
      "paperMentions": {
        "jannach-22-eval-crs": {
          "mark": "check"
        },
        "siro23-crs-satisfaction": {
          "mark": "check",
          "term": "Interest arousal"
        },
        "siro22-tds-satisfaction": {
          "mark": "check",
          "term": "Interest arousal"
        },
        "joko26-face": {
          "mark": "check",
          "term": "Interest Arousal"
        }
      }
    },
    {
      "number": 20,
      "aspect": "User understanding",
      "aliases": [],
      "explanation": {
        "text": "Does the system appear to understand the user throughout the conversation?",
        "citations": [
          {
            "paperId": "mehri20-fed"
          }
        ]
      },
      "paperMentions": {
        "mehri20-fed": {
          "mark": "check",
          "term": "Understanding"
        },
        "siro23-crs-satisfaction": {
          "mark": "check",
          "term": "Understanding"
        },
        "siro22-tds-satisfaction": {
          "mark": "check",
          "term": "Understanding"
        },
        "joko26-face": {
          "mark": "check",
          "term": "Understanding"
        }
      }
    },
    {
      "number": 21,
      "aspect": "Completeness",
      "aliases": [
        "Sufficiency"
      ],
      "explanation": {
        "text": "Does the turn satisfy the requests in the previous user turn?",
        "citations": [
          {
            "paperId": "sakai-23-swan"
          }
        ]
      },
      "paperMentions": {
        "sakai-23-swan": {
          "mark": "check"
        },
        "jannach-22-eval-crs": {
          "mark": "check",
          "term": "Item coverage"
        },
        "xu23-critical-eval": {
          "mark": "check"
        }
      }
    },
    {
      "number": 22,
      "aspect": "Fair treatment",
      "aliases": [
        "Bias"
      ],
      "explanation": {
        "text": "Does the system provide the same benefit to different users and user groups?",
        "citations": [
          {
            "paperId": "sakai-23-swan"
          }
        ]
      },
      "paperMentions": {
        "sakai-23-swan": {
          "mark": "check"
        },
        "zamani22-cis": {
          "mark": "check"
        },
        "milano20-rs-ethics": {
          "mark": "check"
        }
      }
    },
    {
      "number": 23,
      "aspect": "Slot filling",
      "aliases": [],
      "explanation": {
        "text": "Can the system identify terms in statements to fill slots in a structured search query?",
        "citations": [
          {
            "paperId": "zamani22-cis"
          }
        ]
      },
      "paperMentions": {
        "gao22-cir": {
          "mark": "check"
        },
        "zamani22-cis": {
          "mark": "check"
        },
        "deriu20-eval-survey": {
          "mark": "check"
        }
      }
    },
    {
      "number": 24,
      "aspect": "Information need resolution",
      "aliases": [],
      "explanation": {
        "text": "Is the information need ultimately resolved?",
        "citations": [
          {
            "paperId": "zamani22-cis"
          }
        ]
      },
      "paperMentions": {
        "gao22-cir": {
          "mark": "check"
        },
        "zamani22-cis": {
          "mark": "check"
        },
        "jannach-22-eval-crs": {
          "mark": "check",
          "term": "Efectiveness of task support"
        }
      }
    },
    {
      "number": 25,
      "aspect": "Recoverability",
      "aliases": [],
      "explanation": {
        "text": "Does the system turn keep the user interacting after the user has expressed dissatisfaction?",
        "citations": [
          {
            "paperId": "sakai-23-swan"
          }
        ]
      },
      "paperMentions": {
        "sakai-23-swan": {
          "mark": "check"
        },
        "mehri20-fed": {
          "mark": "check",
          "term": "Error recovery"
        }
      }
    },
    {
      "number": 26,
      "aspect": "Adaptability",
      "aliases": [],
      "explanation": {
        "text": "Does the system keep up with the changes in the world?",
        "citations": [
          {
            "paperId": "sakai-23-swan"
          }
        ]
      },
      "paperMentions": {
        "sakai-23-swan": {
          "mark": "check"
        },
        "jannach-22-eval-crs": {
          "mark": "check",
          "term": "Adaptation"
        }
      }
    },
    {
      "number": 27,
      "aspect": "Dialogue act prediction",
      "aliases": [],
      "explanation": {
        "text": "How well does the system predict the dialogue act of a given utterance?",
        "citations": [
          {
            "paperId": "zamani22-cis"
          }
        ]
      },
      "paperMentions": {
        "zamani22-cis": {
          "mark": "check"
        },
        "deriu20-eval-survey": {
          "mark": "check"
        }
      }
    },
    {
      "number": 28,
      "aspect": "Trust",
      "aliases": [],
      "explanation": {
        "text": "Does the user trust the system?",
        "citations": [
          {
            "paperId": "zamani22-cis"
          }
        ]
      },
      "paperMentions": {
        "gao22-cir": {
          "mark": "related"
        },
        "zamani22-cis": {
          "mark": "check"
        }
      }
    },
    {
      "number": 29,
      "aspect": "Cognitive load",
      "aliases": [],
      "explanation": {
        "text": "What is the cognitive load of interactions?",
        "citations": [
          {
            "paperId": "zamani22-cis"
          }
        ]
      },
      "paperMentions": {
        "zamani22-cis": {
          "mark": "check"
        },
        "nvidia-24-nemotron-4": {
          "mark": "check",
          "term": "Verbosity"
        }
      }
    },
    {
      "number": 30,
      "aspect": "Topic depth",
      "aliases": [],
      "explanation": {
        "text": "Can the system sustain a long and cohesive conversation about one topic?",
        "citations": [
          {
            "paperId": "deriu20-eval-survey"
          }
        ]
      },
      "paperMentions": {
        "mehri20-fed": {
          "mark": "check"
        },
        "deriu20-eval-survey": {
          "mark": "check"
        }
      }
    },
    {
      "number": 31,
      "aspect": "Inquisitiveness",
      "aliases": [],
      "explanation": {
        "text": "Does the system demonstrate curiosity by asking questions throughout the conversation?",
        "citations": [
          {
            "paperId": "mehri20-fed"
          }
        ]
      },
      "paperMentions": {
        "gao22-cir": {
          "mark": "check",
          "term": "Clarifying questions"
        },
        "mehri20-fed": {
          "mark": "check"
        }
      }
    },
    {
      "number": 32,
      "aspect": "Mixed initiative",
      "aliases": [],
      "explanation": {
        "text": "The system and user both can take initiative as appropriate.",
        "citations": [
          {
            "paperId": "gao22-cir"
          }
        ]
      },
      "paperMentions": {
        "gao22-cir": {
          "mark": "check"
        },
        "zamani22-cis": {
          "mark": "check"
        }
      }
    },
    {
      "number": 33,
      "aspect": "Privacy",
      "aliases": [],
      "explanation": {
        "text": "Unauthorised data collection and storage, data leaks, and unauthorised inferences.",
        "citations": [
          {
            "paperId": "milano20-rs-ethics"
          }
        ]
      },
      "paperMentions": {
        "gao22-cir": {
          "mark": "check"
        },
        "milano20-rs-ethics": {
          "mark": "check"
        }
      }
    },
    {
      "number": 34,
      "aspect": "Preference",
      "aliases": [],
      "explanation": {
        "text": "Who would you prefer to talk to for a long conversation?  (Also used as engagingness )",
        "citations": [
          {
            "paperId": "smith22-open-problem"
          },
          {
            "paperId": "li19-acute-eval"
          }
        ]
      },
      "paperMentions": {
        "li19-acute-eval": {
          "mark": "check"
        },
        "smith22-open-problem": {
          "mark": "check"
        }
      }
    },
    {
      "number": 35,
      "aspect": "Knowledgeable",
      "aliases": [],
      "explanation": {
        "text": "Who is more knowledgeable?",
        "citations": [
          {
            "paperId": "li19-acute-eval"
          }
        ]
      },
      "paperMentions": {
        "mehri20-fed": {
          "mark": "check",
          "term": "Informative"
        },
        "li19-acute-eval": {
          "mark": "check"
        }
      }
    },
    {
      "number": 36,
      "aspect": "Response specificity",
      "aliases": [],
      "explanation": {
        "text": "Is the response specific to the conversation rather than generic?",
        "citations": [
          {
            "paperId": "mehri20-fed"
          }
        ]
      },
      "paperMentions": {
        "mehri20-fed": {
          "mark": "check",
          "term": "Specific"
        },
        "xu23-critical-eval": {
          "mark": "check"
        }
      }
    },
    {
      "number": 37,
      "aspect": "Sincerity",
      "aliases": [],
      "explanation": {
        "text": "Is the nugget likely to be consistent with the system’s internal results?",
        "citations": [
          {
            "paperId": "sakai-23-swan"
          }
        ]
      },
      "paperMentions": {
        "sakai-23-swan": {
          "mark": "check"
        }
      }
    },
    {
      "number": 38,
      "aspect": "Modesty",
      "aliases": [],
      "explanation": {
        "text": "Does the system’s confidence about the nugget seem appropriate?",
        "citations": [
          {
            "paperId": "sakai-23-swan"
          }
        ]
      },
      "paperMentions": {
        "sakai-23-swan": {
          "mark": "check"
        }
      }
    },
    {
      "number": 39,
      "aspect": "Originality",
      "aliases": [],
      "explanation": {
        "text": "Is the nugget original, and not a copy of some existing text?",
        "citations": [
          {
            "paperId": "sakai-23-swan"
          }
        ]
      },
      "paperMentions": {
        "sakai-23-swan": {
          "mark": "check"
        }
      }
    },
    {
      "number": 40,
      "aspect": "Retentiveness",
      "aliases": [],
      "explanation": {
        "text": "Does the system “remember”?",
        "citations": [
          {
            "paperId": "sakai-23-swan"
          }
        ]
      },
      "paperMentions": {
        "sakai-23-swan": {
          "mark": "check"
        }
      }
    },
    {
      "number": 41,
      "aspect": "Robustness to input variations",
      "aliases": [],
      "explanation": {
        "text": "Does the system eventually provide the same information no matter how we ask?",
        "citations": [
          {
            "paperId": "sakai-23-swan"
          }
        ]
      },
      "paperMentions": {
        "sakai-23-swan": {
          "mark": "check"
        }
      }
    },
    {
      "number": 42,
      "aspect": "User goal prediction",
      "aliases": [],
      "explanation": {
        "text": "How well does the system predict the user’s goals and sub-goals?",
        "citations": [
          {
            "paperId": "zamani22-cis"
          }
        ]
      },
      "paperMentions": {
        "zamani22-cis": {
          "mark": "check"
        }
      }
    },
    {
      "number": 43,
      "aspect": "Scope error",
      "aliases": [],
      "explanation": {
        "text": "Does the response exceed the scope limits of a chatbot?",
        "citations": [
          {
            "paperId": "hosking24-hfg"
          }
        ]
      },
      "paperMentions": {
        "hosking24-hfg": {
          "mark": "check"
        }
      }
    },
    {
      "number": 44,
      "aspect": "Repetition error",
      "aliases": [],
      "explanation": {
        "text": "Does the response repeat itself?",
        "citations": [
          {
            "paperId": "hosking24-hfg"
          }
        ]
      },
      "paperMentions": {
        "hosking24-hfg": {
          "mark": "check"
        }
      }
    },
    {
      "number": 45,
      "aspect": "Refusal error",
      "aliases": [],
      "explanation": {
        "text": "If the request is reasonable, does the response refuse to answer it (e.g. “I’m sorry, I can’t help you with that”)?",
        "citations": [
          {
            "paperId": "hosking24-hfg"
          }
        ]
      },
      "paperMentions": {
        "hosking24-hfg": {
          "mark": "check"
        }
      }
    },
    {
      "number": 46,
      "aspect": "Formatting error",
      "aliases": [],
      "explanation": {
        "text": "Does the response fail to conform to any formatting or length requirements from the prompt?",
        "citations": [
          {
            "paperId": "hosking24-hfg"
          }
        ]
      },
      "paperMentions": {
        "hosking24-hfg": {
          "mark": "check"
        }
      }
    },
    {
      "number": 47,
      "aspect": "Topic breadth",
      "aliases": [],
      "explanation": {
        "text": "Can the system talk about a large variety of topics?",
        "citations": [
          {
            "paperId": "deriu20-eval-survey"
          }
        ]
      },
      "paperMentions": {
        "deriu20-eval-survey": {
          "mark": "check"
        }
      }
    },
    {
      "number": 48,
      "aspect": "Query understanding and augmentation",
      "aliases": [],
      "explanation": {
        "text": "Several components in the CIR system may be used to understand and augment the user’s query. These could include components for named entity recognition, co-reference resolution, query completion and query rewriting.",
        "citations": [
          {
            "paperId": "gao22-cir"
          }
        ]
      },
      "paperMentions": {
        "gao22-cir": {
          "mark": "check"
        }
      }
    },
    {
      "number": 49,
      "aspect": "Content summarization",
      "aliases": [],
      "explanation": {
        "text": "Many IR systems make use of summarization to give users their first view of some content, so they can decide whether to engage further the content.",
        "citations": [
          {
            "paperId": "gao22-cir"
          }
        ]
      },
      "paperMentions": {
        "gao22-cir": {
          "mark": "check"
        }
      }
    },
    {
      "number": 50,
      "aspect": "Elicitation",
      "aliases": [
        "User revealment"
      ],
      "explanation": {
        "text": "The system helps the user express or discover her information need and long-term preferences.",
        "citations": [
          {
            "paperId": "gao22-cir"
          }
        ]
      },
      "paperMentions": {
        "gao22-cir": {
          "mark": "check"
        }
      }
    },
    {
      "number": 51,
      "aspect": "System revealment",
      "aliases": [],
      "explanation": {
        "text": "The system reveals to the user its capabilities and corpus, building the user’s expectations of what it can and cannot do.",
        "citations": [
          {
            "paperId": "gao22-cir"
          }
        ]
      },
      "paperMentions": {
        "gao22-cir": {
          "mark": "check"
        }
      }
    },
    {
      "number": 52,
      "aspect": "Memory",
      "aliases": [],
      "explanation": {
        "text": "The user can reference past statements, which implicitly also remain true unless contradicted.",
        "citations": [
          {
            "paperId": "gao22-cir"
          }
        ]
      },
      "paperMentions": {
        "gao22-cir": {
          "mark": "check"
        }
      }
    },
    {
      "number": 53,
      "aspect": "Set retrieval",
      "aliases": [],
      "explanation": {
        "text": "The system can reason about the utility of sets of complementary items.",
        "citations": [
          {
            "paperId": "gao22-cir"
          }
        ]
      },
      "paperMentions": {
        "gao22-cir": {
          "mark": "check"
        }
      }
    },
    {
      "number": 54,
      "aspect": "Autonomy and personal identity",
      "aliases": [],
      "explanation": {
        "text": "Encroachment on sense of personal identity and behavioural traps.",
        "citations": [
          {
            "paperId": "milano20-rs-ethics"
          }
        ]
      },
      "paperMentions": {
        "milano20-rs-ethics": {
          "mark": "check"
        }
      }
    },
    {
      "number": 55,
      "aspect": "Social efects",
      "aliases": [],
      "explanation": {
        "text": "Lack of exposure to contrasting viewpoints and feedback effects.",
        "citations": [
          {
            "paperId": "milano20-rs-ethics"
          }
        ]
      },
      "paperMentions": {
        "milano20-rs-ethics": {
          "mark": "check"
        }
      }
    },
    {
      "number": 56,
      "aspect": "Preference utilization",
      "aliases": [],
      "explanation": {
        "text": "Can the system effectively utilize the elicitated and stored preferences?",
        "citations": [
          {
            "paperId": "joko24-laps"
          }
        ]
      },
      "paperMentions": {
        "joko24-laps": {
          "mark": "check"
        }
      }
    },
    {
      "number": 57,
      "aspect": "Well-structured",
      "aliases": [],
      "explanation": {
        "text": "",
        "citations": [
          {
            "paperId": "xu23-critical-eval"
          }
        ],
        "note": "This aspect is identified in the paper, but no explicit definition is provided."
      },
      "paperMentions": {
        "xu23-critical-eval": {
          "mark": "check"
        }
      }
    },
    {
      "number": 58,
      "aspect": "Example",
      "aliases": [],
      "explanation": {
        "text": "Provide a clearer example for people who may not have experience in the field.",
        "citations": [
          {
            "paperId": "xu23-critical-eval"
          }
        ]
      },
      "paperMentions": {
        "xu23-critical-eval": {
          "mark": "check"
        }
      }
    },
    {
      "number": 59,
      "aspect": "Helpfulness",
      "aliases": [],
      "explanation": {
        "text": "",
        "citations": [
          {
            "paperId": "nvidia-24-nemotron-4"
          }
        ],
        "note": "This aspect is identified in the paper, but no explicit definition is provided."
      },
      "paperMentions": {
        "nvidia-24-nemotron-4": {
          "mark": "check"
        }
      }
    },
    {
      "number": 60,
      "aspect": "Semantic appropriateness",
      "aliases": [],
      "explanation": {
        "text": "Is the response semantically appropriate within the context of the current conversation?",
        "citations": [
          {
            "paperId": "mehri20-fed"
          }
        ]
      },
      "paperMentions": {
        "mehri20-fed": {
          "mark": "check",
          "term": "Semantically appropriate"
        }
      }
    },
    {
      "number": 61,
      "aspect": "Likeability",
      "aliases": [],
      "explanation": {
        "text": "Does the system display a likeable personality throughout the conversation?",
        "citations": [
          {
            "paperId": "mehri20-fed"
          }
        ]
      },
      "paperMentions": {
        "mehri20-fed": {
          "mark": "check"
        }
      }
    }
  ]
};
