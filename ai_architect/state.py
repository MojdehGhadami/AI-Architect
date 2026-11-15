"""Shared graph state and requirement helpers (no LangGraph imports here)."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, TypedDict

from .config import load_schema

# Fields that must be known before any retrieval tool is called.
CRITICAL_FIELDS = [
    "goal.task_type",
    "data.modality",
    "data.dimensionality",
    "data.dataset_size_tb",
    "training.strategy",
    "training.framework",
    "deployment.environment",
]


class ArchitectState(TypedDict, total=False):
    messages: List[Any]                      # reducer is attached in graph.py
    requirements: Dict[str, Any]
    pending_questions: List[Dict[str, Any]]  # questions waiting for the user
    asked_fields: List[str]
    clarify_rounds: int
    retrieval_plan: Optional[Dict[str, Any]]
    evidence: Dict[str, Any]
    final_response: Optional[Dict[str, Any]]


def empty_requirements() -> Dict[str, Any]:
    return copy.deepcopy(load_schema("requirements.template"))


def is_unknown(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "unknown"}
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False


def get_path(obj: Dict[str, Any], path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def missing_critical(requirements: Dict[str, Any]) -> List[str]:
    return [p for p in CRITICAL_FIELDS if is_unknown(get_path(requirements, p))]


def merge_requirements(current: Dict[str, Any], update: Any) -> Dict[str, Any]:
    """Merge an extractor update into the current requirements.

    Only keys that exist in the template are kept, and a known value is
    never overwritten by an unknown one. decision_gates is always
    recomputed here rather than trusted from the model.
    """
    merged = copy.deepcopy(current) if current else empty_requirements()
    update = update if isinstance(update, dict) else {}

    for section, fields in empty_requirements().items():
        if section == "decision_gates" or not isinstance(fields, dict):
            continue
        new_section = update.get(section)
        if not isinstance(new_section, dict):
            continue
        for key in fields:
            new_value = new_section.get(key)
            if not is_unknown(new_value):
                merged[section][key] = new_value

    missing = missing_critical(merged)
    merged["decision_gates"] = {"tool_ready": not missing, "missing_critical": missing}
    return merged


def role_and_content(message: Any) -> tuple[str, str]:
    """Read a message whether it is a dict or a LangChain message object."""
    if isinstance(message, dict):
        role, content = message.get("role", ""), message.get("content", "")
    else:
        role, content = getattr(message, "type", ""), getattr(message, "content", "")
    role = {"human": "user", "ai": "assistant"}.get(role, role)
    return role, content if isinstance(content, str) else str(content)


def transcript(messages: List[Any], limit: int = 20) -> List[Dict[str, str]]:
    out = []
    for m in (messages or [])[-limit:]:
        role, content = role_and_content(m)
        if role in {"user", "assistant"} and content:
            out.append({"role": role, "content": content})
    return out
