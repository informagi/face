#!/usr/bin/env python3
"""Single-utterance nuggetization pipeline backed by OpenRouter."""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Protocol, Tuple

from dotenv import load_dotenv
import requests
from tqdm import tqdm

# Load .env file if present
load_dotenv()

# Local dependencies
from deps.prompts_particle_generation import prompt_template
from deps.json_extraction import extract_json
from deps.particle_tuple_selector import PluralityVoteHandler
# Dry run import
from deps.llm_dry_run_particle_generator import get_dry_run_particles
CURR_DIR = Path(__file__).resolve().parent


VALID_DIALOGUE_ACTS = {
    "greeting",
    "preference elicitation",
    "recommendation",
    "goodbye",
    "others",
}

DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct"
DEFAULT_SYSTEM_PROMPT = "You are a professional assistant who strictly follows the instructions."


class OpenRouterClient:
    """Lightweight client for OpenRouter chat completions."""

    api_url = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.6,
        max_tokens: int = 2048,
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
                "OPENROUTER_API_KEY environment variable is required for OpenRouter requests."
            )

    def complete(self, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": os.getenv("OPENROUTER_REFERRER", "https://conv-eval.local"),
            "X-Title": os.getenv("OPENROUTER_TITLE", "FACE Particle Generator"),
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
            self.api_url,
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
        # CURR_DIR is face/particle_generation
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



class ParticleValidator:
    """Ensures LLM responses follow the expected particle schema."""

    required_keys = {"dialogue_act", "particle", "user_feedback"}

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, file=sys.stderr)

    def validate(self, payload: Dict) -> bool:
        if not isinstance(payload, dict):
            self._log("Error: Response is not a dictionary")
            return False
        
        # Support both new and legacy keys for intermediate robustness, but internal logic prefers particle
        if "particle_generation_results" in payload:
            item_list = payload["particle_generation_results"]
        elif "nuggetization_results" in payload:
            item_list = payload["nuggetization_results"]
        else:
            self._log("Error: Missing 'particle_generation_results' key")
            return False

        if not isinstance(item_list, list):
            self._log("Error: Results is not a list")
            return False
        if len(item_list) == 0:
            return True

        for item in item_list:
            if not isinstance(item, dict):
                self._log("Error: Particle entry is not a dictionary")
                return False
            
            # Map legacy 'nugget' key to 'particle' if needed for validation
            if "nugget" in item and "particle" not in item:
                item["particle"] = item["nugget"]

            if not self.required_keys.issubset(item.keys()):
                missing = self.required_keys - item.keys()
                self._log(f"Error: Missing required keys {missing}")
                return False
            if item["dialogue_act"] not in VALID_DIALOGUE_ACTS:
                self._log(f"Error: Invalid dialogue_act '{item['dialogue_act']}'")
                return False
            if not isinstance(item["particle"], str):
                self._log("Error: particle is not a string")
                return False
            word_count = len(item["particle"].split())
            if not (1 <= word_count <= 12):
                self._log(
                    f"Error: particle must be 1-12 words, got {word_count} words"
                )
                return False
            if not isinstance(item["user_feedback"], str) or item["user_feedback"] == "None":
                self._log("Error: user_feedback is not a string or is 'None'")
                return False
        return True


class ParticlePromptBuilder:
    """Builds prompts that mirror the original particle generation setup."""

    def __init__(self, template: str = prompt_template) -> None:
        self.template = template

    def build(
        self,
        utterance: str,
        speaker_label: str = "USER",
        dialogue_history: str = "(No dialogue history provided.)",
        user_response: str = "(No subsequent turns provided.)",
    ) -> str:
        utterance_norm = " ".join(utterance.strip().split())
        target_turn = f"Turn 0 - {speaker_label}: {utterance_norm}"
        return self.template.format(
            dialogue_history=dialogue_history,
            target_turn=target_turn,
            user_response=user_response,
        )


class ParticleGenerator:
    """Orchestrates prompt creation, sampling, validation, and voting."""

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        n_samples: int = 20,
        max_attempts: int = 100,
        verbose: bool = False,
        dry_run: bool = False,
    ) -> None:
        self.client = client if client is not None else (None if dry_run else OpenRouterClient())
        self.n_samples = n_samples
        self.max_attempts = max_attempts
        self.prompt_builder = ParticlePromptBuilder()
        self.validator = ParticleValidator(verbose=verbose)
        self.verbose = verbose
        self.dry_run = dry_run

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, file=sys.stderr)

    def _get_valid_responses(self, prompt: str) -> List[List[Dict]]:
        prompts = [prompt] * self.n_samples

        if self.dry_run:
            raw_responses = get_dry_run_particles(prompts)
            valid_responses = []
            for raw in raw_responses:
                # Still use extract_json to simulate realistic parsing pipeline
                parsed = extract_json(raw, flag_longest=True)
                if parsed and self.validator.validate(parsed):
                    # Handle both keys for compatibility
                    if "particle_generation_results" in parsed:
                        valid_responses.append(parsed["particle_generation_results"])
                    else:
                        valid_responses.append(parsed["nuggetization_results"])
                else:
                    self._log("Dry run generated invalid response (unexpected).")
            return valid_responses

        responses: List[Optional[List[Dict]]] = [None] * len(prompts)
        incomplete_indices = list(range(len(prompts)))
        progress_bar = tqdm(
            total=self.n_samples,
            desc="Collecting particles",
            unit="completion",
        )

        if self.client is None:
            raise RuntimeError("LLM client is required unless --dry-run is enabled.")

        try:
            max_workers = min(len(prompts), os.cpu_count() or 4)
            if max_workers == 0:
                max_workers = 1

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for attempt in range(self.max_attempts):
                    remaining = len(incomplete_indices)
                    self._log(
                        f"Retry {attempt}: {remaining} prompts remaining"
                    )
                    if remaining == 0:
                        break

                    future_to_idx = {
                        executor.submit(self.client.complete, prompts[idx]): idx
                        for idx in incomplete_indices
                        if responses[idx] is None
                    }

                    next_round: List[int] = []
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        if responses[idx] is not None:
                            continue
                        try:
                            completion = future.result()
                        except requests.RequestException as exc:
                            self._log(f"OpenRouter request failed: {exc}")
                            next_round.append(idx)
                            continue
                        except Exception as exc:
                            self._log(f"Unexpected error during completion: {exc}")
                            next_round.append(idx)
                            continue

                        parsed = extract_json(completion, flag_longest=True)
                        if parsed and self.validator.validate(parsed):
                            if "particle_generation_results" in parsed:
                                responses[idx] = parsed["particle_generation_results"]
                            else:
                                responses[idx] = parsed["nuggetization_results"]
                            progress_bar.update(1)
                        else:
                            self._log("Invalid response encountered; retrying.")
                            next_round.append(idx)

                    incomplete_indices = [i for i in next_round if responses[i] is None]
        finally:
            progress_bar.close()

        valid_responses = [resp for resp in responses if resp is not None]
        if not valid_responses:
            raise RuntimeError(
                "Failed to obtain any valid particle generation responses from the model."
            )
        return valid_responses

    def generate_particles(
        self,
        utterance: str,
        speaker_label: str = "USER",
        dialogue_history: str = "(No dialogue history provided.)",
        user_response: str = "(No subsequent turns provided.)",
    ) -> List[Dict]:
        prompt = self.prompt_builder.build(
            utterance,
            speaker_label=speaker_label,
            dialogue_history=dialogue_history,
            user_response=user_response,
        )
        responses = self._get_valid_responses(prompt)
        vote_handler = PluralityVoteHandler(responses)
        selected, _ = vote_handler.get_highest_voted_response()
        return selected["content"]


def generate_particles_from_utterance(
    utterance: str,
    n_samples: int = 20,
    max_attempts: int = 100,
    speaker_label: str = "USER",
    temperature: float = 0.6,
    max_tokens: int = 2048,
    verbose: bool = False,
    dry_run: bool = False,
    dialogue_history: str = "(No dialogue history provided.)",
    user_response: str = "(No subsequent turns provided.)",
) -> List[Dict]:

    client = OpenRouterClient(
        temperature=temperature,
        max_tokens=max_tokens,
    )
    generator = ParticleGenerator(
        client=client,
        n_samples=n_samples,
        max_attempts=max_attempts,
        verbose=verbose,
        dry_run=dry_run,
    )

    return generator.generate_particles(
        utterance,
        speaker_label=speaker_label,
        dialogue_history=dialogue_history,
        user_response=user_response,
    )


def extract_context(
    dialogue: List[Dict], target_idx: int
) -> Tuple[str, str, str]:
    """Extract history, target turn, and user feedback from dialogue."""
    if not (0 <= target_idx < len(dialogue)):
        raise ValueError(f"Target index {target_idx} out of bounds.")

    # 1. Dialogue History
    if target_idx == 0:
        history_str = "(No dialogue history provided.)"
    else:
        history_lines = []
        for i in range(target_idx):
            turn = dialogue[i]
            role = turn.get("role", "UNKNOWN")
            content = turn.get("utterance", "") or turn.get("content", "")
            history_lines.append(f"Turn {i} - {role}: {content}")
        history_str = "\n".join(history_lines)

    # 2. Target Turn
    target_turn = dialogue[target_idx]
    content = target_turn.get("utterance", "") or target_turn.get("content", "")

    # 3. User Feedback (Next Turn)
    next_idx = target_idx + 1
    if next_idx < len(dialogue):
        feedback_turn = dialogue[next_idx]
        feedback_content = feedback_turn.get("utterance", "") or feedback_turn.get("content", "")
        feedback_str = feedback_content if feedback_content.strip() else "(No subsequent turns provided.)"
    else:
        feedback_str = "(No subsequent turns provided.)"

    return history_str, content, feedback_str


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate particles for a single utterance.")
    parser.add_argument(
        "utterance",
        type=str,

        help="The utterance text to process OR path to a conversation JSON file.",
    )
    parser.add_argument(
        "--turn-index",
        type=int,
        default=-1,
        help="If input is a file, specify the turn index to process (default: last assistant entry).",
    )
    parser.add_argument("--speaker", default="USER", help="Speaker label used in the prompt.")
    parser.add_argument("--samples", type=int, default=20, help="Number of valid samples to collect.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=100,
        help="Maximum completion attempts before giving up.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature for the OpenRouter model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate per completion.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save the particle generation results as JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print validation debug information to stderr.",
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

    args = parser.parse_args()

    # Select LLM client: custom if provided, otherwise OpenRouter
    if args.dry_run:
        client = None
    elif args.custom_llm:
        client = load_custom_llm(
            args.custom_llm,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    else:
        client = OpenRouterClient(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    generator = ParticleGenerator(
        client=client,
        n_samples=args.samples,
        max_attempts=args.max_attempts,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )


    # Determine input type
    input_arg = args.utterance
    possible_file = Path(input_arg)
    
    final_utterance = input_arg
    final_history = "(No dialogue history provided.)"
    final_feedback = "(No subsequent turns provided.)"
    
    if possible_file.exists() and possible_file.is_file() and input_arg.endswith(".json"):
        if args.verbose:
            print(f"Loading conversation from {possible_file}", file=sys.stderr)
        try:
            dialogue = json.loads(possible_file.read_text())
            if isinstance(dialogue, list):
                # Determine target index
                if args.turn_index == -1:
                    # Auto-detect last assistant turn
                    candidates = [
                        i for i, t in enumerate(dialogue) 
                        if t.get("role") == "ASST" or t.get("role") == "assistant"
                    ]
                    if not candidates:
                        raise ValueError("No assistant turns found in dialogue.")
                    target_idx = candidates[-1]
                else:
                    target_idx = args.turn_index
                
                final_history, final_utterance, final_feedback = extract_context(dialogue, target_idx)
                
                if args.verbose:
                    print(f"Targeting turn {target_idx}", file=sys.stderr)
                    print(f"History len: {len(final_history)} chars", file=sys.stderr)
                    print(f"Feedback: {final_feedback[:50]}...", file=sys.stderr)

        except Exception as e:
            print(f"Error reading conversation file: {e}", file=sys.stderr)
            sys.exit(1)

    particles = generator.generate_particles(
        final_utterance,
        speaker_label=args.speaker,
        dialogue_history=final_history,
        user_response=final_feedback,
    )
    output_payload = json.dumps(particles, indent=2)
    print(output_payload)
    
    if args.output:
        output_path = args.output
    else:
        # Default to ../results/particle_<hash_or_timestamp>.json
        import time
        timestamp = int(time.time())
        results_dir = CURR_DIR.parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        # Sanitize utterance for filename or just use timestamp
        safe_utterance = "".join(x for x in args.utterance if x.isalnum())[:20]
        output_path = results_dir / f"particle_{timestamp}_{safe_utterance}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_payload)
    print(f"Results saved to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
