"""
agents/llm.py

LLM client using Ollama. No API key required.
Defaults to tinyllama but respects the OLLAMA_MODEL env var.

Ollama must be running locally:
    ollama serve
    ollama pull tinyllama

Set OLLAMA_HOST in .env to point at a remote Ollama instance.
"""

import json
import os
import re

import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama")

# TinyLlama has a 2048 token context window.
# Keep prompts short and responses minimal.
MAX_PROMPT_CHARS = 1500
NUM_PREDICT = 512


def call_ollama(prompt: str, system: str = "") -> str:
    """Call Ollama and return raw text response."""
    url = f"{OLLAMA_HOST}/api/generate"

    # Truncate prompt hard if needed to fit context
    combined = f"{system}\n\n{prompt}" if system else prompt
    if len(combined) > MAX_PROMPT_CHARS:
        combined = combined[:MAX_PROMPT_CHARS]

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": combined,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": NUM_PREDICT,
            "num_ctx": 2048,
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=180)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_HOST}. "
            "Run: ollama serve"
        )
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Ollama request failed: {e}")

    return response.json().get("response", "").strip()


def repair_json(raw: str) -> str:
    """
    Attempt to repair truncated JSON from TinyLlama.
    Closes any unclosed braces/brackets so json.loads has a chance.
    """
    raw = raw.strip()

    # Strip markdown fences
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)

    # Try to find the first { ... } block even if truncated
    start = raw.find('{')
    if start == -1:
        return raw
    raw = raw[start:]

    # Count open braces/brackets and close them
    open_braces = 0
    open_brackets = 0
    in_string = False
    escape_next = False

    for ch in raw:
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            open_braces += 1
        elif ch == '}':
            open_braces -= 1
        elif ch == '[':
            open_brackets += 1
        elif ch == ']':
            open_brackets -= 1

    # Close any unclosed string first
    if in_string:
        raw += '"'

    # Close unclosed arrays then objects
    raw += ']' * max(0, open_brackets)
    raw += '}' * max(0, open_braces)

    return raw


def call_llm_json(prompt: str, system: str = "") -> dict:
    """
    Call Ollama and parse response as JSON.
    Repairs truncated JSON before giving up.
    """
    raw = call_ollama(prompt, system=system)

    repaired = repair_json(raw)

    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        # Final fallback: return safe minimal dict
        return {
            "answer": raw[:500],
            "citations": [],
            "statute_refs": [],
            "reasoning": "",
            "extraction_confidence": 0.1,
            "extraction_notes": f"JSON parse failed after repair attempt. Raw: {raw[:200]}",
        }


def check_ollama_available() -> tuple[bool, str]:
    """Check whether Ollama is reachable and the model is available."""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        response.raise_for_status()
        models = [m["name"] for m in response.json().get("models", [])]
        model_base = OLLAMA_MODEL.split(":")[0]
        available = any(model_base in m for m in models)
        if not available:
            return False, (
                f"Model '{OLLAMA_MODEL}' not found. "
                f"Run: ollama pull {OLLAMA_MODEL}\n"
                f"Available: {models}"
            )
        return True, f"Ollama OK. Model: {OLLAMA_MODEL}"
    except Exception as e:
        return False, f"Ollama not reachable at {OLLAMA_HOST}: {e}"
