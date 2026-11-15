"""Retrieval Planner Agent: decides which knowledge bases to query and how."""

from __future__ import annotations

from typing import Any, Callable, Dict

from ..config import load_prompt, load_schema
from ..llm_json import LLMJsonError, call_llm_json
from ..state import ArchitectState, missing_critical


def blocked_plan(reason: str) -> Dict[str, Any]:
    return {"should_retrieve": False, "blocked_reason": reason, "queries": []}


def make_planner_node(llm: Any) -> Callable[[ArchitectState], Dict[str, Any]]:
    system = load_prompt("retrieval_planner")
    schema = load_schema("retrieval_plan")

    def planner_node(state: ArchitectState) -> Dict[str, Any]:
        requirements = state.get("requirements") or {}

        # Hard gate in code, not only in the prompt: never retrieve on guesses.
        missing = missing_critical(requirements)
        if missing:
            return {"retrieval_plan": blocked_plan("Missing critical requirements: " + ", ".join(missing))}

        try:
            plan = call_llm_json(llm, system, {"requirements": requirements}, schema)
        except LLMJsonError as exc:
            plan = blocked_plan(f"Retrieval planner failed to produce a valid plan: {exc}")

        return {"retrieval_plan": plan}

    return planner_node
