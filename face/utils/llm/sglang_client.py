"""Ready-to-use SGLang Client for FACE.

This client connects to a local SGLang server running at http://localhost:30000.

Usage:
    uv run face.py ... --custom-llm sglang_client.py
"""

import requests
from typing import Optional


class CustomLLM:
    """SGLang-based LLM client."""
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.6,
        max_tokens: int = 2048,
        endpoint: str = "http://localhost:30000/generate",
        **kwargs
    ) -> None:
        """Initialize SGLang client.
        
        Args:
            model: Not used by SGLang client (server determines model), but accepted for compatibility.
            temperature: Sampling temperature.
            max_tokens: Max new tokens to generate.
            endpoint: URL of the SGLang server (default: http://localhost:30000/generate).
            **kwargs: Extra arguments ignored.
        """
        self.model = model or "sglang-custom"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.endpoint = endpoint
    
    def complete(self, prompt: str) -> str:
        """Send prompt to SGLang server and return generated text."""
        response = requests.post(
            self.endpoint,
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": self.temperature,
                    "max_new_tokens": self.max_tokens,
                }
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["text"]
