"""Requirements Extractor Agent: conversation -> structured requirements."""

from __future__ import annotations

from typing import Any, Callable, Dict

from ..config import load_prompt, load_schema
from ..llm_json import LLMJsonError, call_llm_json
from ..state import ArchitectState, empty_requirements, merge_requirements, transcript

MAX_MESSAGE_CHARS = 1500


def make_extractor_node(llm: Any) -> Callable[[ArchitectState], Dict[str, Any]]:
    system = load_prompt("requirements_extractor")
    schema = load_schema("requirements.schema")

    def extractor_node(state: ArchitectState) -> Dict[str, Any]:
        current = state.get("requirements") or merge_requirements(empty_requirements(), {})
        conversation = [
            {"role": m["role"], "content": m["content"][:MAX_MESSAGE_CHARS]}
            for m in transcript(state.get("messages", []))
        ]

        try:
            requirements = call_llm_json(
                llm,
                system,
                {"current_requirements": current, "conversation": conversation},
                schema,
                # Merge before validating: keeps known values, drops unknown
                # keys, recomputes decision_gates deterministically.
                postprocess=lambda update: merge_requirements(current, update),
            )
        except LLMJsonError:
            requirements = current  # keep what we had; the clarifier will re-ask

        # A new user turn invalidates any previous plan/evidence/answer.
        return {
            "requirements": requirements,
            "pending_questions": [],
            "retrieval_plan": None,
            "evidence": {},
            "final_response": None,
        }

    return extractor_node
