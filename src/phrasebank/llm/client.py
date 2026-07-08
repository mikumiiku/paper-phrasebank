"""OpenAI-compatible LLM client wrapper with retry + JSON-sandwich robustness."""
from __future__ import annotations

import json
import re
from typing import Any

import openai
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)


class LLMError(Exception):
    """Raised on non-recoverable LLM transport or parsing failures."""


# Retry only on transport / 5xx errors.  4xx and JSON parse failures surface
# immediately (first-fault) — retrying them would be wasted latency.
_RETRYABLE = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.APIStatusError,
)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, openai.APIStatusError):
        return 500 <= exc.status_code < 600
    return isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError))


class LLMClient:
    """Thin wrapper around ``openai.OpenAI`` exposing ``call_json``.

    Parameters
    ----------
    client: openai.OpenAI
        The underlying SDK client (dependency-injected).
    model_name:
        Model id override.
    """

    def __init__(self, client: openai.OpenAI, model_name: str) -> None:
        self._c = client
        self._model = model_name

    @property
    def raw(self) -> openai.OpenAI:
        return self._c

    def _invoke(self, system: str, user: str, as_object: bool) -> str:
        return _chat_completion(
            client=self._c,
            model=self._model,
            system=system,
            user=user,
            as_object=as_object,
        )

    def call_json(self, system: str, user: str) -> list[dict[str, Any]]:
        """Sentence extraction entry point — returns a list of objects.

        We request ``response_format=json_object`` so the model emits a top-level
        object (reliable across providers); the dict-unwrap fallback then expands
        the first list-valued field (e.g. ``{"sentences": [...]}``).  If a model
        still returns a bare ``[...]``, the sandwich parser handles that too.
        """
        text = self._invoke(system, user, as_object=True)
        return _parse_as_array(text)

    def call_object(self, system: str, user: str) -> dict[str, Any]:
        """Metadata extraction entry point — returns a single dict."""
        text = self._invoke(system, user, as_object=True)
        return _parse_as_object(text)


# ──────────────────────────────────────────────────────────────────────────────
# Transport
# ──────────────────────────────────────────────────────────────────────────────


def _chat_completion(
    client: openai.OpenAI, model: str, system: str, user: str, as_object: bool, **kwargs: Any
) -> str:
    # One retry (2 attempts total) on network / 5xx.
    do = retry(
        reraise=True,
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception(_is_retryable),
    )
    try:
        resp = do(_raw_create)(client, model, system, user, as_object=as_object, **kwargs)
    except openai.APIStatusError as exc:
        raise LLMError(f"LLM API status error {exc.status_code}: {exc.message}") from exc
    except openai.OpenAIError as exc:
        raise LLMError(f"LLM transport error: {exc}") from exc

    content = resp.choices[0].message.content
    if content is None:
        raise LLMError("LLM returned empty message content")
    return content


def _raw_create(
    client: openai.OpenAI, model: str, system: str, user: str, as_object: bool, **kwargs: Any
) -> Any:
    rf: dict[str, str] | None
    if as_object:
        rf = {"type": "json_object"}
    else:
        rf = None
    return client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format=rf,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **kwargs,
    )


# ──────────────────────────────────────────────────────────────────────────────
# JSON sandwich parser
# ──────────────────────────────────────────────────────────────────────────────

_OPENER_RE = re.compile(r"[\[{]")


def _sandwich(text: str) -> Any:
    """Forced sandwich: first ``[``/``{`` ... last ``]``/``}``."""
    first = _OPENER_RE.search(text)
    if first is None:
        raise LLMError("LLM response contains no JSON structure:\n" + text[:500])
    start = first.start()
    # Match the correct closer to the opener.
    opener = text[start]
    closer = "]" if opener == "[" else "}"
    end = text.rfind(closer)
    if end < start:
        raise LLMError(
            f"LLM response missing closing '{closer}' for opener '{opener}':\n" + text[:500]
        )
    payload = text[start : end + 1]
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise LLMError(f"JSON parse failed after sandwich: {exc}\nPayload: {payload[:500]}") from exc


def _parse_as_array(text: str) -> list[dict[str, Any]]:
    data = _sandwich(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Only allowed fallback: dict wrapping an array ({"sentences": [...]}).
        for v in data.values():
            if isinstance(v, list):
                return v
        raise LLMError(
            "LLM returned a dict with no list-valued field to unwrap:\n"
            + json.dumps(data, ensure_ascii=False)[:500]
        )
    raise LLMError(f"LLM returned unexpected JSON type {type(data).__name__}:\n{str(data)[:500]}")


def _parse_as_object(text: str) -> dict[str, Any]:
    data = _sandwich(text)
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        # e.g. [{"paper_title": ..., ...}] — take the first object.
        for item in data:
            if isinstance(item, dict):
                return item
        raise LLMError("LLM returned a list with no object entry to unwrap")
    raise LLMError(f"LLM returned unexpected JSON type {type(data).__name__}:\n{str(data)[:500]}")
