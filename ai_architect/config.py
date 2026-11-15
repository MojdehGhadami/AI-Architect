"""Configuration, prompt/schema loading and LLM construction.

All secrets come from environment variables (see .env.example).
Nothing in this module needs an API key at import time.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = PACKAGE_DIR / "prompts"
SCHEMAS_DIR = PACKAGE_DIR / "schemas"

# How many clarification turns before the system stops asking and reports
# what is still missing instead.
MAX_CLARIFY_ROUNDS = int(os.getenv("MAX_CLARIFY_ROUNDS", "3"))


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def load_schema(name: str) -> dict:
    return json.loads((SCHEMAS_DIR / f"{name}.json").read_text(encoding="utf-8"))


def get_database_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is not set. Copy .env.example to .env and fill it in.")
    return url


def get_llm():
    """Build the chat model selected by LLM_PROVIDER (mistral | openai).

    API keys are read by the client libraries from MISTRAL_API_KEY /
    OPENAI_API_KEY.
    """
    provider = os.getenv("LLM_PROVIDER", "mistral").strip().lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0)

    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI

        return ChatMistralAI(model=os.getenv("MISTRAL_MODEL", "mistral-large-latest"), temperature=0)

    raise ValueError(f"Unsupported LLM_PROVIDER '{provider}'. Use 'mistral' or 'openai'.")
