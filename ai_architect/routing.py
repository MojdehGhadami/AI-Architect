"""Conditional-edge functions (kept LangGraph-free so they are unit-testable)."""

from __future__ import annotations

from .state import ArchitectState


def route_after_clarifier(state: ArchitectState) -> str:
    """Stop and wait for the user if the clarifier asked something."""
    return "ask_user" if state.get("pending_questions") else "planner"


def route_after_planner(state: ArchitectState) -> str:
    """Skip retrieval when the planner blocked it; the final agent reports why."""
    plan = state.get("retrieval_plan") or {}
    return "retriever" if plan.get("should_retrieve") else "final"
