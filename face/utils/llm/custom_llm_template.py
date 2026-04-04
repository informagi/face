"""Custom LLM Client Template for FACE

This module provides a template for implementing custom LLM clients
that can be used with FACE scoring and particle generation tools.

To use a custom LLM (e.g., a local SGLang server), create a Python file
implementing the `CustomLLM` class with a `complete(prompt: str) -> str` method.

Example usage:
    uv run face.py --conversation conv.json --aspect dialogue_overall \
        --custom-llm my_sglang_client.py
"""

from typing import Optional


class CustomLLM:
    """Base template for custom LLM clients.
    
    Your custom implementation must have:
    1. An __init__ method that accepts **kwargs (for forward compatibility)
    2. A `complete(prompt: str) -> str` method that returns the LLM response
    
    The FACE tools will instantiate your class and call the `complete` method.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.6,
        max_tokens: int = 2048,
        **kwargs
    ) -> None:
        """Initialize the custom LLM client.
        
        Args:
            model: Model identifier (interpretation depends on implementation)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments for forward compatibility
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def complete(self, prompt: str) -> str:
        """Generate a completion for the given prompt.
        
        Args:
            prompt: The user prompt to complete
            
        Returns:
            The generated text response from the LLM
            
        Raises:
            NotImplementedError: This is a template - implement in subclass
        """
        raise NotImplementedError(
            "Subclass must implement the `complete` method. "
            "See face/docs/llm_setup.md for implementation guide."
        )


# =============================================================================
# EXAMPLE: SGLang Implementation
# =============================================================================

class SGLangClient(CustomLLM):
    """Example implementation using a local SGLang server.
    
    This example assumes you have an SGLang server running locally.
    Adjust the endpoint and request format based on your setup.
    
    SGLang Server Setup:
        python -m sglang.launch_server --model-path <model> --port 30000
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.6,
        max_tokens: int = 2048,
        endpoint: str = "http://localhost:30000/generate",
        **kwargs
    ) -> None:
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.endpoint = endpoint
        
        # Import requests here to make the base template dependency-free
        try:
            import requests
            self._requests = requests
        except ImportError:
            raise ImportError(
                "requests library required for SGLangClient. "
                "Install with: pip install requests"
            )
    
    def complete(self, prompt: str) -> str:
        """Send completion request to local SGLang server."""
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": self.temperature,
                "max_new_tokens": self.max_tokens,
            }
        }
        
        response = self._requests.post(
            self.endpoint,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        
        data = response.json()
        # SGLang returns {"text": "..."} by default
        return data.get("text", "")


# =============================================================================
# For testing: A dummy client that returns mock responses
# =============================================================================

class DummyLLM(CustomLLM):
    """A dummy LLM for testing the custom LLM loading mechanism."""
    
    def complete(self, prompt: str) -> str:
        """Return a mock JSON response for testing."""
        # Return a valid FACE scoring response format
        return '{"dialogue_overall_score": 3, "reasoning": "Dummy response"}'
