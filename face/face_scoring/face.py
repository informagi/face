#!/usr/bin/env python3
"""FACE: Fine-grained Aspect-based Conversation Evaluator.

This script runs inference-only particle evaluation using the top-16 FACE prompts.
Given a dialogue and an evaluation aspect, it scores each particle in assistant
turns by sampling multiple LLM completions per prompt, then aggregates results.

Usage:
    uv run face.py \\
        --conversation test_conv.json \\
        --aspect dialogue_overall \\
        --output results.json
"""

from __future__ import annotations

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library
import argparse
import importlib.util
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

# Third-party
import requests
from dotenv import load_dotenv
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader

# Load environment variables before local imports
load_dotenv()

# Local dependencies
CURR_DIR = Path(__file__).resolve().parent


def clean_utterance(text: str) -> str:
    """Normalize whitespace in utterance text."""
    return " ".join(text.replace("\n", " ").split())


JINJA_ENV = Environment(loader=FileSystemLoader(CURR_DIR / "prompt_templates"))
JINJA_ENV.filters["clean_utterance"] = clean_utterance
sys.path.append(str(CURR_DIR / "deps"))

import aspect_utils  # noqa: E402
from json_extraction import extract_json  # noqa: E402
# Dry run import
from deps.llm_dry_run_scoring import get_dry_run_scores


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct"
DEFAULT_SYSTEM_PROMPT = "You are a professional assistant who strictly follows the instructions."
DEFAULT_SAMPLING_NUM = 5
DEFAULT_NUM_TRY = 10
DEFAULT_TEMPERATURE = 0.6


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PromptInfo:
    """Metadata for a single evaluation prompt."""
    filename: str
    text: str


@dataclass
class CallSpec:
    """Specification for a single LLM call."""
    prompt_index: int
    prompt_info: PromptInfo
    conv_id: str
    turn_index: int
    turn_index: int
    particle_index: int
    sampling_index: int
    prompt_text: str

    @property
    def instance_id(self) -> str:
        """Unique ID for a prompt-particle pair (excludes sampling index)."""
        return (
            f"prompt_ind:{self.prompt_index}--conv_id:{self.conv_id}--"
            f"turn_ind:{self.turn_index}--particle_ind:{self.particle_index}"
        )


# =============================================================================
# OPENROUTER CLIENT
# =============================================================================

class OpenRouterClient:
    """Minimal client for OpenRouter chat completions."""

    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = 768,
        timeout: int = 60,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.system_prompt = system_prompt
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY environment variable is required."
            )

    def complete(self, prompt: str) -> str:
        """Send a completion request and return the response text."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": os.getenv("OPENROUTER_REFERRER", "https://conv-eval.local"),
            "X-Title": os.getenv("OPENROUTER_TITLE", "FACE Evaluator"),
        }
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        response = requests.post(
            self.API_URL,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices")
        if not choices:
            raise RuntimeError("OpenRouter response did not include any choices.")
        return choices[0]["message"]["content"]


class LLMClient(Protocol):
    """Protocol defining the interface for LLM clients."""
    
    def complete(self, prompt: str) -> str:
        """Generate a completion for the given prompt."""
        ...


def load_custom_llm(path: str, **kwargs) -> LLMClient:
    """Dynamically load a custom LLM client from a Python file.
    
    Args:
        path: Path to the Python file containing a CustomLLM class
        **kwargs: Arguments to pass to the CustomLLM constructor
        
    Returns:
        An instance of the CustomLLM class from the specified file
    """
    # resolve path: check exact, then utils/llm
    candidate_path = Path(path)
    if not candidate_path.exists():
        # Try face/utils/llm/ as fallback
        repo_utils = CURR_DIR.parent / "utils" / "llm"
        
        candidates = [
            repo_utils / path,
            repo_utils / f"{path}.py"
        ]
        
        for cand in candidates:
            if cand.exists():
                candidate_path = cand
                break
                
    spec = importlib.util.spec_from_file_location("custom_llm", str(candidate_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path} (resolved to {candidate_path})")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Look for CustomLLM class first, then SGLangClient as fallback
    if hasattr(module, "CustomLLM"):
        return module.CustomLLM(**kwargs)
    elif hasattr(module, "SGLangClient"):
        return module.SGLangClient(**kwargs)
    else:
        raise AttributeError(
            f"Module {path} must define a 'CustomLLM' or 'SGLangClient' class"
        )


# =============================================================================
# FACE EVALUATOR
# =============================================================================

class FaceEvaluator:
    """Particle-level dialogue evaluator using FACE prompts."""

    def __init__(
        self,
        aspect: str,
        prompts: Sequence[PromptInfo],
        client: Optional[LLMClient],
        sampling_num: int = DEFAULT_SAMPLING_NUM,
        num_try: int = DEFAULT_NUM_TRY,
        include_null: bool = False,
        max_workers: Optional[int] = None,
        verbose: bool = False,
        dry_run: bool = False,
    ) -> None:
        if aspect not in aspect_utils.aspect_to_level:
            raise ValueError(
                f"Unknown aspect '{aspect}'. "
                f"Expected one of {sorted(aspect_utils.aspect_to_level)}."
            )
        self.aspect = aspect
        self.prompts = list(prompts)
        self.client = client
        self.model_name = getattr(client, "model", "dry-run")
        self.sampling_num = sampling_num
        self.num_try = num_try
        self.include_null = include_null
        self.max_workers = max_workers
        self.verbose = verbose
        self.dry_run = dry_run

        # Load aspect-specific configuration
        self.evaluation_level = aspect_utils.aspect_to_level[aspect]
        self.aspect_description = aspect_utils.aspect_dict_nugget_based[aspect]
        self.grading_description = aspect_utils.get_grading_description_string_nugget_based(
            aspect, include_null=include_null
        )
        self.pred_score_info = aspect_utils.pred_score_info[aspect]
        self.score_key = f"{aspect}_score"
        self.fallback_score = self.pred_score_info["fall_back_for_all_non_applicable_turn"]

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def evaluate(
        self,
        dialogue: Sequence[Dict[str, Any]],
        conv_id: str,
        max_workers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluate all particles in a dialogue and return aggregated results."""
        calls = self._prepare_calls(dialogue, conv_id)
        if not calls:
            raise ValueError("No particles found in the dialogue for evaluation.")

        worker_count = max_workers or self.max_workers or min(8, len(calls)) or 1
        responses = self._execute_calls(calls, worker_count)
        return self._aggregate(dialogue, conv_id, calls, responses)

    # -------------------------------------------------------------------------
    # Prompt Construction
    # -------------------------------------------------------------------------

    def _build_prompt_text(
        self,
        instruction: str,
        dialogue: Sequence[Dict[str, Any]],
        turn_index: int,
        particle: Dict[str, Any],
    ) -> str:
        """Build the full prompt text for evaluating a particle."""
        if self.evaluation_level == "turn_level":
            return self._build_turn_level_prompt(instruction, dialogue, turn_index, particle)
        return self._build_dialogue_level_prompt(instruction, dialogue, turn_index, particle)

    def _build_turn_level_prompt(
        self,
        instruction: str,
        dialogue: Sequence[Dict[str, Any]],
        turn_index: int,
        particle: Dict[str, Any],
    ) -> str:
        """Build prompt for turn-level aspects (relevance, interestingness)."""
        template = JINJA_ENV.get_template("turn_level.jinja2")
        return template.render(
            instruction_prompt=instruction,
            dialogue=dialogue,
            turn_index=turn_index,
            target_nugget=str(particle),
            aspect=self.aspect,
            aspect_description=self.aspect_description,
            grading_description_string=self.grading_description,
            min_score=self.pred_score_info["min_score"],
            max_score=self.pred_score_info["max_score"],
        )

    def _build_dialogue_level_prompt(
        self,
        instruction: str,
        dialogue: Sequence[Dict[str, Any]],
        turn_index: int,
        particle: Dict[str, Any],
    ) -> str:
        """Build prompt for dialogue-level aspects."""
        template = JINJA_ENV.get_template("dialogue_level.jinja2")
        return template.render(
            instruction_prompt=instruction,
            dialogue=dialogue,
            target_turn_ind=turn_index,
            target_nugget=str(particle),
            aspect=self.aspect,
            aspect_description=self.aspect_description,
            grading_description_string=self.grading_description,
            min_score=self.pred_score_info["min_score"],
            max_score=self.pred_score_info["max_score"],
        )

    # -------------------------------------------------------------------------
    # LLM Call Execution
    # -------------------------------------------------------------------------

    def _prepare_calls(
        self,
        dialogue: Sequence[Dict[str, Any]],
        conv_id: str,
    ) -> List[CallSpec]:
        """Generate all call specifications for the dialogue."""
        calls: List[CallSpec] = []
        for prompt_idx, prompt_info in enumerate(self.prompts):
            for turn in dialogue:
                if turn.get("role") != "ASST":
                    continue
                # Support both keys
                particles = turn.get("particles") or turn.get("nuggets") or []
                if not particles:
                    continue
                turn_idx = turn["turn_ind"]
                for particle_idx, particle in enumerate(particles):
                    prompt_text = self._build_prompt_text(
                        prompt_info.text, dialogue, turn_idx, particle
                    )
                    for sample_idx in range(self.sampling_num):
                        calls.append(CallSpec(
                            prompt_index=prompt_idx,
                            prompt_info=prompt_info,
                            conv_id=conv_id,
                            turn_index=turn_idx,
                            particle_index=particle_idx,
                            sampling_index=sample_idx,
                            prompt_text=prompt_text,
                        ))
        return calls

    def _execute_calls(
        self,
        calls: List[CallSpec],
        worker_count: int,
    ) -> List[Dict[str, Any]]:
        """Execute all LLM calls in parallel with progress bar."""
        
        if self.dry_run:
            # Batch dry run
            prompts = [c.prompt_text for c in calls]
            # Since dry run is local and fast, we can just process sequentially or simple batch
            raw_responses = get_dry_run_scores(prompts, self.aspect)
            responses = []
            for raw in raw_responses:
                 # Construct response dict matching _call_with_retries structure
                 responses.append({
                     "score_json": raw,
                     "raw_completion": raw,
                     "attempts": 1,
                     "errors": [],
                 })
            return responses

        responses: List[Optional[Dict[str, Any]]] = [None] * len(calls)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(self._call_with_retries, call.prompt_text): idx
                for idx, call in enumerate(calls)
            }
            with tqdm(total=len(calls), desc="Scoring particles", unit="call") as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    responses[idx] = future.result()
                    pbar.update(1)

        return responses

    def _call_with_retries(self, prompt_text: str) -> Dict[str, Any]:
        """Make an LLM call with retry logic for JSON extraction failures."""
        if self.client is None:
            raise RuntimeError("LLM client is required unless --dry-run is enabled.")

        errors: List[str] = []

        for attempt in range(1, self.num_try + 1):
            try:
                completion = self.client.complete(prompt_text)
            except requests.RequestException as exc:
                errors.append(f"request_error:{exc}")
                continue
            except Exception as exc:
                errors.append(f"unexpected_error:{exc}")
                continue

            extracted = extract_json(completion, flag_longest=True)
            if extracted and self._validate_score(extracted):
                processed = self._postprocess_score(extracted)
                return {
                    "score_json": json.dumps(processed),
                    "raw_completion": completion,
                    "attempts": attempt,
                    "errors": errors,
                }
            errors.append("invalid_json")

        # All retries failed - use fallback
        fallback = self._postprocess_score({
            self.score_key: self.pred_score_info["min_score"]
        })
        return {
            "score_json": json.dumps(fallback),
            "raw_completion": None,
            "attempts": self.num_try,
            "errors": errors,
            "used_fallback": True,
        }

    def _validate_score(self, extracted: Dict[str, Any]) -> bool:
        """Validate that extracted JSON contains a valid score."""
        if extracted is None:
            return False

        if self.score_key not in extracted:
            if self.verbose:
                print(f"Missing key '{self.score_key}' in response {extracted}")
            return False

        score = extracted[self.score_key]
        if score is None:
            return self.include_null

        try:
            score_int = int(score)
        except (TypeError, ValueError):
            if self.verbose:
                print(f"Score value not convertible to int: {score}")
            return False

        min_score = self.pred_score_info["min_score"]
        max_score = self.pred_score_info["max_score"]
        if not (min_score <= score_int <= max_score):
            if self.verbose:
                print(f"Score {score_int} outside range {min_score}-{max_score}")
            return False

        return True

    def _postprocess_score(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add weight_score field based on score value."""
        score = data.get(self.score_key)
        if score is None:
            data[self.score_key] = None
            data["weight_score"] = 0.0
        else:
            data[self.score_key] = int(score)
            if data[self.score_key] == self.pred_score_info["non_applicable"]:
                data["weight_score"] = 0.0
            else:
                data["weight_score"] = 1.0
        return data

    # -------------------------------------------------------------------------
    # Result Aggregation
    # -------------------------------------------------------------------------

    def _aggregate(
        self,
        dialogue: Sequence[Dict[str, Any]],
        conv_id: str,
        calls: Sequence[CallSpec],
        responses: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate all responses into final results structure."""
        # Step 1: Group by instance (prompt + particle)
        by_instance = self._group_by_instance(calls, responses)

        # Step 2: Aggregate per prompt
        per_prompt = self._aggregate_per_prompt(by_instance)

        # Step 3: Build prompt results list
        prompt_results = self._build_prompt_results(per_prompt)

        # Step 4: Calculate ensemble statistics
        ensemble = self._calculate_ensemble(prompt_results)

        # Step 5: Aggregate across all prompts per particle
        per_particle = self._aggregate_across_prompts(prompt_results)

        return {
            "conversation_id": conv_id,
            "aspect": self.aspect,
            "evaluation_level": self.evaluation_level,
            "model": self.model_name,
            "sampling_num": self.sampling_num,
            "num_try": self.num_try,
            "num_prompts": len(self.prompts),
            "score_range": {
                "min": self.pred_score_info["min_score"],
                "max": self.pred_score_info["max_score"],
            },
            "prompt_results": prompt_results,
            "ensemble": ensemble,
            "per_particle_ensemble": per_particle,
        }

    def _group_by_instance(
        self,
        calls: Sequence[CallSpec],
        responses: Sequence[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Group responses by prompt-particle instance."""
        by_instance: Dict[str, Dict[str, Any]] = {}

        for call, response in zip(calls, responses):
            if response is None:
                raise RuntimeError("Missing response for a call specification.")

            entry = by_instance.setdefault(call.instance_id, {
                "prompt_index": call.prompt_index,
                "prompt_file": call.prompt_info.filename,
                "turn_index": call.turn_index,
                "particle_index": call.particle_index,
                "scores": [],
                "weights": [],
                "raw_samples": [],
            })

            score_json = response.get("score_json")
            if score_json:
                parsed = json.loads(score_json)
                entry["raw_samples"].append(parsed)
                entry["scores"].append(parsed.get(self.score_key))
                entry["weights"].append(parsed.get("weight_score"))

        return by_instance

    def _aggregate_per_prompt(
        self,
        by_instance: Dict[str, Dict[str, Any]],
    ) -> Dict[int, Dict[str, Any]]:
        """Aggregate particle scores for each prompt."""
        per_prompt: Dict[int, Dict[str, Any]] = {}

        for info in by_instance.values():
            prompt_idx = info["prompt_index"]
            entry = per_prompt.setdefault(prompt_idx, {
                "prompt_file": info["prompt_file"],
                "particle_breakdown": [],
            })

            valid_scores = [s for s in info["scores"] if s is not None]
            avg_score = mean(valid_scores) if valid_scores else None
            avg_weight = (
                mean([w for w in info["weights"] if w is not None])
                if any(w is not None for w in info["weights"])
                else 0.0
            )

            entry["particle_breakdown"].append({
                "turn_index": info["turn_index"],
                "particle_index": info["particle_index"],
                "avg_score": avg_score,
                "avg_weight": avg_weight,
                "samples": info["raw_samples"],
            })

        # Sort particles and calculate overall score per prompt
        for prompt_idx, entry in per_prompt.items():
            entry["particle_breakdown"].sort(
                key=lambda x: (x["turn_index"], x["particle_index"])
            )
            scores = [n["avg_score"] for n in entry["particle_breakdown"]]
            weights = [n["avg_weight"] for n in entry["particle_breakdown"]]
            entry["overall_score"] = self._weighted_average(scores, weights)
            entry["prompt_text"] = self.prompts[prompt_idx].text

        return per_prompt

    def _build_prompt_results(
        self,
        per_prompt: Dict[int, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build the final prompt_results list."""
        return [
            {
                "rank": idx + 1,
                "prompt_file": per_prompt[i]["prompt_file"],
                "overall_score": per_prompt[i]["overall_score"],
                "particle_breakdown": per_prompt[i]["particle_breakdown"],
            }
            for idx, i in enumerate(sorted(per_prompt))
        ]

    def _calculate_ensemble(
        self,
        prompt_results: List[Dict[str, Any]],
    ) -> Dict[str, Optional[float]]:
        """Calculate ensemble statistics across all prompts."""
        scores = [
            r["overall_score"]
            for r in prompt_results
            if r["overall_score"] is not None
        ]
        if not scores:
            return {"mean": None, "median": None, "min": None, "max": None}
        return {
            "mean": mean(scores),
            "median": median(scores),
            "min": min(scores),
            "max": max(scores),
        }

    def _aggregate_across_prompts(
        self,
        prompt_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Aggregate scores across all prompts for each particle."""
        scores_by_particle: Dict[Tuple[int, int], List[float]] = {}

        for entry in prompt_results:
            for particle in entry["particle_breakdown"]:
                key = (particle["turn_index"], particle["particle_index"])
                if particle["avg_score"] is not None:
                    scores_by_particle.setdefault(key, []).append(particle["avg_score"])

        return [
            {
                "turn_index": turn_idx,
                "particle_index": particle_idx,
                "mean_score": mean(scores),
                "median_score": median(scores),
                "num_votes": len(scores),
            }
            for (turn_idx, particle_idx), scores in sorted(scores_by_particle.items())
        ]

    def _weighted_average(
        self,
        scores: Sequence[Optional[float]],
        weights: Sequence[Optional[float]],
    ) -> Optional[float]:
        """Calculate weighted average of scores."""
        valid_pairs = [
            (s, w if w is not None else 0.0)
            for s, w in zip(scores, weights)
            if s is not None
        ]
        if not valid_pairs:
            return None

        total_weight = sum(w for _, w in valid_pairs)
        if total_weight == 0:
            return float(self.fallback_score)

        return sum(s * w for s, w in valid_pairs) / total_weight


# =============================================================================
# FILE I/O UTILITIES
# =============================================================================

def load_prompts(prompts_path: Path) -> List[PromptInfo]:
    """Load prompts from a JSON file."""
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")
        
    data = json.loads(prompts_path.read_text())
    return [
        PromptInfo(filename=item["filename"], text=item["text"])
        for item in data
    ]


def load_dialogue(path: Path) -> List[Dict[str, Any]]:
    """Load and validate a dialogue from a JSON file."""
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Conversation file must contain a list of turns.")

    dialogue: List[Dict[str, Any]] = []
    for idx, turn in enumerate(data):
        if not isinstance(turn, dict):
            raise ValueError(f"Turn {idx} is not a dictionary: {turn}")
        role = turn.get("role")
        utterance = turn.get("utterance")
        if role is None or utterance is None:
            raise ValueError(f"Turn {idx} missing 'role' or 'utterance'.")
        dialogue.append({
            "turn_ind": idx,
            "role": role,
            "utterance": utterance,
            "utterance": utterance,
            # Maintain backward compatibility in loading but use 'particles' internally
            "particles": turn.get("particles") or turn.get("nuggets") or [],
        })
    return dialogue


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run FACE evaluation on a dialogue.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--conversation",
        type=Path,
        required=True,
        help="Path to the JSON conversation file.",
    )
    parser.add_argument(
        "--aspect",
        type=str,
        required=True,
        choices=sorted(aspect_utils.aspect_to_level.keys()),
        help="Evaluation aspect.",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="Path to prompt JSON file. Defaults to top_16_prompts/<aspect>.json.",
    )
    parser.add_argument(
        "--conv-id",
        type=str,
        default="face_conv",
        help="Identifier used in instance IDs.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLING_NUM,
        help="Number of samples per prompt/particle pair.",
    )
    parser.add_argument(
        "--num-try",
        type=int,
        default=DEFAULT_NUM_TRY,
        help="Maximum retries per completion.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="OpenRouter model to use.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum parallel completions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save the evaluation results as JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for JSON validation.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Limit the number of prompts (for testing).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode with deterministic, mock outputs.",
    )
    parser.add_argument(
        "--custom-llm",
        type=str,
        default=None,
        help="Path to a custom LLM client Python file. See face/docs/llm_setup.md for details.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Main entry point."""
    args = parse_args(argv)

    # Set default prompts file
    if args.prompts_file is None:
        args.prompts_file = CURR_DIR / "top_16_prompts" / f"{args.aspect}.json"

    # Load data
    prompts = load_prompts(args.prompts_file)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    dialogue = load_dialogue(args.conversation)

    # Select LLM client: custom if provided, otherwise OpenRouter
    if args.dry_run:
        client = None
    elif args.custom_llm:
        client = load_custom_llm(
            args.custom_llm,
            model=args.model,
            temperature=args.temperature,
        )
    else:
        client = OpenRouterClient(
            model=args.model,
            temperature=args.temperature,
        )
    evaluator = FaceEvaluator(
        aspect=args.aspect,
        prompts=prompts,
        client=client,
        sampling_num=args.samples,
        num_try=args.num_try,
        max_workers=args.max_workers,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )

    # Run evaluation
    results = evaluator.evaluate(dialogue, conv_id=args.conv_id)

    # Output results
    output_payload = json.dumps(results, indent=2)
    print(output_payload)

    if args.output:
        output_path = args.output
    else:
        # Default to ../results/<conv_id>_<aspect>.json
        results_dir = CURR_DIR.parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f"{args.conv_id}_{args.aspect}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_payload)
    print(f"Results saved to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
