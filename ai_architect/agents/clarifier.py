"""Clarifying Agent: decides which questions (if any) to ask the user."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

from ..config import MAX_CLARIFY_ROUNDS, load_prompt, load_schema
from ..llm_json import LLMJsonError, call_llm_json
from ..state import ArchitectState, get_path, is_unknown, missing_critical

# Used only if the model fails to produce valid questions for missing fields.
FALLBACK_QUESTIONS: Dict[str, Dict[str, Any]] = {
    "goal.task_type": {
        "question": "What should the model do with your data (e.g. classify, detect, segment, generate text)?",
        "why_it_matters": "The task decides which model families and how much GPU memory are needed.",
    },
    "data.modality": {
        "question": "What kind of data do you have?",
        "why_it_matters": "Different data types need different models and storage.",
        "options": ["image", "text", "audio", "tabular", "multimodal"],
    },
    "data.dimensionality": {
        "question": "Is your data 1D (signals/text), 2D (images) or 3D (volumes/video)?",
        "why_it_matters": "3D data needs much more GPU memory than 2D.",
        "options": ["1D", "2D", "3D", "not_applicable"],
    },
    "data.dataset_size_tb": {
        "question": "Approximately how large is your dataset (in GB or TB)?",
        "why_it_matters": "Dataset size drives storage, training time and GPU count.",
    },
    "training.strategy": {
        "question": "Will you train from scratch, fine-tune a pre-trained model, or only run inference?",
        "why_it_matters": "This determines the GPU memory tier, GPU count and cost.",
        "options": ["train_from_scratch", "fine_tune", "inference_only"],
    },
    "training.framework": {
        "question": "Do you have a framework preference?",
        "why_it_matters": "It affects the training stack and serving options.",
        "options": ["pytorch", "tensorflow", "jax", "no_preference"],
    },
    "deployment.environment": {
        "question": "Where should this run?",
        "why_it_matters": "It decides which infrastructure options are feasible.",
        "options": ["cloud_k8s", "on_prem", "hybrid", "edge_device", "air_gapped"],
    },
}


def _fallback(missing: List[str]) -> List[Dict[str, Any]]:
    return [{"field": f, **FALLBACK_QUESTIONS[f]} for f in missing[:3]]


def make_clarifier_node(llm: Any) -> Callable[[ArchitectState], Dict[str, Any]]:
    system = load_prompt("clarifier")
    schema = load_schema("clarifying")

    def clarifier_node(state: ArchitectState) -> Dict[str, Any]:
        requirements = state.get("requirements") or {}
        rounds = state.get("clarify_rounds", 0)
        asked = list(state.get("asked_fields") or [])
        missing = missing_critical(requirements)

        # Question budget spent: stop asking, let the planner report the gap.
        if rounds >= MAX_CLARIFY_ROUNDS:
            return {"pending_questions": []}

        allow_optional = rounds == 0 and "deployment.compliance" not in asked
        if not missing and not allow_optional:
            return {"pending_questions": []}

        try:
            result = call_llm_json(
                llm,
                system,
                {
                    "requirements": requirements,
                    "missing_critical": missing,
                    "already_asked": asked,
                    "allow_optional_question": allow_optional,
                },
                schema,
            )
            questions = result["questions"]
        except LLMJsonError:
            questions = []

        # Guardrails: only ask about fields that really are unknown.
        questions = [q for q in questions if is_unknown(get_path(requirements, q["field"]))]
        if not allow_optional:
            questions = [q for q in questions if q["field"] in missing]
        if missing and not any(q["field"] in missing for q in questions):
            questions = _fallback(missing)

        if not questions:
            return {"pending_questions": []}

        return {
            "pending_questions": questions,
            "asked_fields": sorted(set(asked) | {q["field"] for q in questions}),
            "clarify_rounds": rounds + 1,
            "messages": [
                {"role": "assistant", "content": json.dumps({"questions": questions}, indent=2, ensure_ascii=False)}
            ],
        }

    return clarifier_node
