# LLM Setup Guide

FACE supports two options for LLM integration: **OpenRouter** (cloud-based) and **custom local models** (e.g., SGLang).

---

## Option 1: OpenRouter (Recommended for Quick Start)

[OpenRouter](https://openrouter.ai/) provides access to various LLMs through a unified API.

### Setup

1. Create an account at [openrouter.ai](https://openrouter.ai/)
2. Generate an API key from the dashboard
3. Create a `.env` file in the repository root:
   ```env
   OPENROUTER_API_KEY=your_key_here
   ```

### Usage

OpenRouter is the default backend. Simply run the tools:

```bash
# Particle generation
uv run particle_generator.py "Your utterance here"

# FACE scoring
uv run face.py --conversation conv.json --aspect dialogue_overall
```

Use `--model` to specify a different model:
```bash
uv run face.py --conversation conv.json --aspect dialogue_overall \
    --model "meta-llama/llama-3.1-8b-instruct"
```

---

## Option 2: Custom LLM (SGLang / Local Models)

For local inference or custom model servers, implement a custom LLM client.

### Step 1: Install and Start SGLang Server

SGLang is a high-performance serving framework for LLMs. Install using `uv` or `pip`:

```bash
pip install uv
uv pip install "sglang"
```

Start the server with your model:

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 30000
```

> **Note**: Adjust `--model-path` to your model. For Docker-based deployment, see the [SGLang documentation](https://github.com/sgl-project/sglang).

Verify the server is running:
```bash
curl http://localhost:30000/health
```

### Step 2: Run FACE with Ready-to-Use Client

We provide a pre-configured client file [`face/utils/llm/sglang_client.py`](../utils/llm/sglang_client.py) that works out-of-the-box.

Simply pass it to the tools using `--custom-llm` (specify the filename (without extension) if in `face/utils/llm/`, or the full path otherwise):

```bash
# Particle generation
uv run particle_generator.py "Your utterance here" \
    --custom-llm sglang_client

# FACE scoring
uv run face.py --conversation conv.json --aspect dialogue_overall \
    --custom-llm sglang_client
```

### Advanced: Custom Implementation

If you need a different client setup (e.g., non-standard endpoint, authentication, or different backend), create a new Python file implementing the `CustomLLM` class.

**Requirements:**
- Class must be named `CustomLLM`
- Must have `__init__(self, **kwargs)`
- Must have `complete(self, prompt: str) -> str`

**Example:**
See [`face/utils/llm/sglang_client.py`](../utils/llm/sglang_client.py) or [`face/utils/llm/custom_llm_template.py`](../utils/llm/custom_llm_template.py) for reference implementations.

---

## Comparison

| Feature | OpenRouter | Custom LLM |
|---------|------------|------------|
| Setup | API key only | Server setup required |
| Cost | Pay per token | Hardware/compute costs |
| Latency | Network dependent | Low (local) |
| Models | 100+ models | Your choice |
| Privacy | Data sent to cloud | Fully local |
