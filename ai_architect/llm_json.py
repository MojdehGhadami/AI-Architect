"""Schema-validated JSON calls to a chat model, with self-repair retries."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional

from jsonschema import validate

STRICT_SUFFIX = (
    "\n\nCRITICAL OUTPUT RULES:\n"
    "- Output ONLY valid JSON. No markdown, no code blocks, no wrappers.\n"
    "- Do not add any keys not defined in the schema.\n"
    "- Do not explain or comment.\n"
)


class LLMJsonError(ValueError):
    """The model did not produce schema-valid JSON within the retry budget."""


def extract_json(text: str) -> str:
    """Pull the first JSON object/array out of a model reply."""
    text = (text or "").strip()

    fence = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()

    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                _, end = decoder.raw_decode(text[i:])
                return text[i : i + end]
            except json.JSONDecodeError:
                continue
    return text


def _content(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):  # some providers return content blocks
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content or "")


def call_llm_json(
    llm: Any,
    system_prompt: str,
    user_content: Any,
    schema: dict,
    max_retries: int = 2,
    postprocess: Optional[Callable[[Any], Any]] = None,
) -> Any:
    """Ask `llm` for JSON that validates against `schema`.

    On a parse or validation failure the error is fed back to the model and
    it gets another attempt, up to `max_retries` times. `postprocess` runs
    before validation (e.g. to merge with existing state).
    """
    if not isinstance(user_content, str):
        user_content = json.dumps(user_content, ensure_ascii=False, indent=2)

    system = system_prompt + "\n\nJSON SCHEMA:\n" + json.dumps(schema, ensure_ascii=False) + STRICT_SUFFIX
    prompt = user_content
    last_text, last_error = "", ""

    for _ in range(max_retries + 1):
        response = llm.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
        )
        last_text = _content(response).strip()

        try:
            data = json.loads(extract_json(last_text))
            if postprocess is not None:
                data = postprocess(data)
            validate(instance=data, schema=schema)
            return data
        except Exception as exc:  # JSONDecodeError, ValidationError, postprocess errors
            last_error = str(exc)[:1200]
            prompt = (
                "Your previous output did not match the schema.\n"
                "Fix it to match EXACTLY. Output JSON only.\n\n"
                f"VALIDATION ERROR:\n{last_error}\n\n"
                f"YOUR PREVIOUS OUTPUT:\n{last_text[:4000]}\n\n"
                f"ORIGINAL INPUT:\n{user_content}"
            )

    raise LLMJsonError(f"No schema-valid JSON after {max_retries + 1} attempts. Last error: {last_error}")
