"""Google Gemini client, built on the current ``google-genai`` SDK.

The deprecated ``google-generativeai`` package is deliberately avoided. This
module is the single place the rest of the app talks to an LLM, so swapping
providers later touches one file.

Every public function is safe to call with no API key: it raises
:class:`LLMError`, which callers catch to fall back to non-LLM behaviour. This
matters because schema detection is Gemini-first — the fallback path is not
optional decoration, it is how the app runs offline.
"""

from __future__ import annotations

import json
import re
from typing import Any

from core.config import settings
from utils.helpers import LLMError
from utils.logger import get_logger

logger = get_logger(__name__)

_client: Any = None


def _runtime_key() -> str:
    """Resolve the active API key: a key set at runtime in the Streamlit
    session takes precedence over the one from the environment/.env, so the
    Settings page can inject a key without a restart."""
    try:
        import streamlit as st

        key = st.session_state.get("gemini_api_key")
        if key:
            return str(key).strip()
    except Exception:  # noqa: BLE001 - not running under Streamlit
        pass
    return settings.llm.api_key.strip()


def is_available() -> bool:
    """True when a key is configured (runtime or env) and the SDK imports."""
    if not _runtime_key():
        return False
    try:
        from google import genai  # noqa: F401
    except ImportError:
        return False
    return True


def _get_client() -> Any:
    """Lazily construct the genai client using the active key."""
    global _client
    key = _runtime_key()
    if _client is None:
        if not key:
            raise LLMError(
                "GEMINI_API_KEY is not set. Add it to your .env file to enable "
                "AI-driven schema detection, the Business Copilot, document chat, "
                "and the written report narrative. Without it the app falls back "
                "to offline heuristics."
            )
        try:
            from google import genai

            _client = genai.Client(api_key=key)
        except ImportError as exc:
            raise LLMError("google-genai is not installed.") from exc
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Failed to initialise Gemini client: {exc}") from exc
    return _client


def reset_client() -> None:
    """Drop the cached client (call after changing the key at runtime)."""
    global _client
    _client = None
    logger.info("Gemini client reset")


def generate(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float | None = None,
    json_mode: bool = False,
) -> str:
    """Run a single-turn generation and return the text.

    Args:
        prompt: The user prompt.
        system: Optional system instruction.
        temperature: Override the configured temperature.
        json_mode: Ask the model to return JSON (response_mime_type).

    Raises:
        LLMError: On any failure, including a missing key.
    """
    from google.genai import types

    client = _get_client()
    cfg: dict[str, Any] = {
        "temperature": settings.llm.temperature if temperature is None else temperature,
        "max_output_tokens": settings.llm.max_output_tokens,
    }
    if system:
        cfg["system_instruction"] = system
    if json_mode:
        cfg["response_mime_type"] = "application/json"

    try:
        response = client.models.generate_content(
            model=settings.llm.model,
            contents=prompt,
            config=types.GenerateContentConfig(**cfg),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Gemini generation failed")
        raise LLMError(str(exc)) from exc

    text = getattr(response, "text", None)
    if not text:
        raise LLMError("Gemini returned an empty response.")
    return text.strip()


def generate_json(prompt: str, *, system: str | None = None) -> Any:
    """Generate and parse a JSON response, tolerating code-fence wrapping."""
    raw = generate(prompt, system=system, json_mode=True, temperature=0.1)
    return _parse_json(raw)


def _parse_json(raw: str) -> Any:
    """Parse JSON that may be wrapped in ```json fences or have prose around it."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # last resort: grab the outermost {...} or [...]
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError as exc:
                raise LLMError(f"Could not parse JSON from model output: {exc}") from exc
        raise LLMError("Model did not return valid JSON.")


__all__ = ["is_available", "generate", "generate_json", "reset_client", "_get_client"]
