"""Final Architect Agent: evidence -> grounded architecture recommendation.

The model writes the recommendation; code then checks every GPU and
Kubernetes component against tool evidence, overwrites numeric facts with
the evidence values, and blocks the answer if nothing grounded is left.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from jsonschema import ValidationError, validate

from ..config import load_prompt, load_schema
from ..llm_json import LLMJsonError, call_llm_json
from ..state import ArchitectState

NOT_VERIFIED = "expert baseline / estimate (not tool-verified)"


def blocked_response(reason: str, assumptions: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "ready_to_answer": False,
        "blocked_reason": reason,
        "assumptions": assumptions or [],
        "gpu_recommendations": [],
        "model_recommendations": [],
        "k8s_components": [],
        "architecture_diagram_mermaid": "",
        "deployment_plan": [],
        "cost_estimate": {
            "method": "Cost cannot be computed because grounded infrastructure evidence is missing.",
            "rough_range_usd": "not available",
        },
    }


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def ground_gpu_recommendations(recs: List[Dict[str, Any]], infra: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only instances present in evidence; copy hard facts from evidence."""
    by_key = {(_norm(r.get("provider")), _norm(r.get("instance_type"))): r for r in infra}
    known_types = [str(r.get("instance_type")) for r in infra if r.get("instance_type")]

    grounded = []
    for rec in recs:
        row = by_key.get((_norm(rec.get("provider")), _norm(rec.get("instance_type"))))
        if row is None:
            continue  # not in evidence -> hallucinated -> dropped

        gpu_count = max(1, int(row.get("gpu_count") or 1))
        vram = row.get("gpu_vram_per_gpu_gb") or (float(row.get("gpu_vram_total_gb") or 0) / gpu_count)
        price = row.get("price_per_hour_instance")

        fallbacks = [f for f in rec.get("fallbacks", []) if _norm(f) in {_norm(t) for t in known_types}]
        if not fallbacks:
            fallbacks = [t for t in known_types if _norm(t) != _norm(row.get("instance_type"))][:2]
        if not fallbacks:
            fallbacks = ["no alternative in retrieved evidence"]

        grounded.append(
            {
                **rec,
                "provider": str(row.get("provider")),
                "instance_type": str(row.get("instance_type")),
                "gpu_type": str(row.get("gpu_model") or rec.get("gpu_type") or "unknown"),
                "gpu_count": gpu_count,
                "vram_gb": max(1.0, round(float(vram or 1), 2)),
                "price_per_hour": float(price) if price else None,
                "fallbacks": fallbacks,
            }
        )
    return grounded


def ground_k8s_components(components: List[Dict[str, Any]], packages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    names = {_norm(p.get("name")): str(p.get("name")) for p in packages if p.get("name")}

    grounded = []
    for comp in components:
        primary = names.get(_norm(comp.get("primary")))
        if primary is None:
            continue
        alternatives = [names[_norm(a)] for a in comp.get("alternatives", []) if _norm(a) in names and _norm(a) != _norm(primary)]
        grounded.append({**comp, "primary": primary, "alternatives": alternatives or ["none in retrieved evidence"]})
    return grounded


def ground_model_recommendations(recs: List[Dict[str, Any]], models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Models may come from expert reasoning, but must be labelled honestly."""
    by_id = {_norm(m.get("model_id")): m for m in models}

    out = []
    for rec in recs:
        row = by_id.get(_norm(rec.get("model_id")))
        if row is not None:
            out.append({**rec, "model_id": str(row["model_id"]), "license": str(row.get("license") or "unknown")})
        else:
            notes = rec.get("notes", "")
            if NOT_VERIFIED not in notes:
                notes = f"{notes} [{NOT_VERIFIED}]".strip()
            out.append({**rec, "license": "unknown", "params_billion": None, "notes": notes})
    return out


def make_final_node(llm: Any) -> Callable[[ArchitectState], Dict[str, Any]]:
    system = load_prompt("final_response")
    schema = load_schema("final_response")

    def _reply(response: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "final_response": response,
            "messages": [{"role": "assistant", "content": json.dumps(response, indent=2, ensure_ascii=False)}],
        }

    def final_node(state: ArchitectState) -> Dict[str, Any]:
        requirements = state.get("requirements") or {}
        plan = state.get("retrieval_plan") or {}
        evidence = state.get("evidence") or {}
        infra = evidence.get("infrastructure") or []
        packages = evidence.get("packages") or []
        models = evidence.get("models") or []
        errors = [f"{e.get('kb')}: {e.get('error')}" for e in evidence.get("errors") or []]

        # Gate 1: the planner refused to retrieve.
        if not plan.get("should_retrieve"):
            return _reply(blocked_response(plan.get("blocked_reason") or "Retrieval was not performed."))

        # Gate 2: no infrastructure evidence -> no LLM call, no guessing.
        if not infra or not packages:
            what = " and ".join(n for n, rows in (("GPU/cloud pricing", infra), ("Kubernetes component", packages)) if not rows)
            return _reply(blocked_response(f"Tool evidence missing: no {what} results were retrieved.", errors))

        try:
            response = call_llm_json(
                llm,
                system,
                {
                    "requirements": requirements,
                    "tool_evidence": {"infrastructure": infra, "packages": packages, "models": models},
                    "evidence_errors": errors,
                },
                schema,
            )
        except LLMJsonError as exc:
            return _reply(blocked_response("Final response could not be produced as valid JSON.", [str(exc)]))

        if not response["ready_to_answer"]:
            return _reply(response)  # schema already guarantees the blocked shape

        # Deterministic grounding.
        response["gpu_recommendations"] = ground_gpu_recommendations(response["gpu_recommendations"], infra)
        response["k8s_components"] = ground_k8s_components(response["k8s_components"], packages)
        response["model_recommendations"] = ground_model_recommendations(response["model_recommendations"], models)

        if not response["gpu_recommendations"] or not response["k8s_components"]:
            return _reply(
                blocked_response(
                    "The drafted recommendation did not match retrieved evidence and was discarded.",
                    ["GPU instances and Kubernetes components must appear in tool evidence."],
                )
            )

        response["assumptions"].append(
            "GPU type, count, VRAM and hourly price are copied from cloud_pricing_db evidence, not generated."
        )
        suspicious = {
            str(r.get("instance_type")) for r in infra if r.get("price_flag") == "suspicious"
        } & {g["instance_type"] for g in response["gpu_recommendations"]}
        for inst in sorted(suspicious):
            response["assumptions"].append(f"Price for {inst} looks unusually low in the pricing feed; verify before budgeting.")

        try:
            validate(instance=response, schema=schema)
        except ValidationError as exc:
            return _reply(blocked_response("Grounded response failed schema validation.", [exc.message]))

        return _reply(response)

    return final_node
