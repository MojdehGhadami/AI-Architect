"""Retriever: executes the planner's RetrievalPlan against the tools.

This node has no LLM. Which tools run, and with which arguments, is decided
entirely by the Retrieval Planner Agent's plan.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping

from ..state import ArchitectState

CLOUD_FILTER_KEYS = {"min_cpu", "min_ram_gb", "min_vram_gb", "provider", "preference"}

# kb name in the plan -> evidence bucket
BUCKETS = {
    "cloud_pricing_db": "infrastructure",
    "k8s_marketplace": "packages",
    "hf_models": "models",
}


def _row_key(bucket: str, row: Dict[str, Any]) -> tuple:
    if bucket == "infrastructure":
        return (row.get("provider"), row.get("instance_type"), row.get("gpu_model"))
    if bucket == "packages":
        return (row.get("name"),)
    return (row.get("model_id"),)


def make_retriever_node(tools: Mapping[str, Callable[..., Dict[str, Any]]]) -> Callable[[ArchitectState], Dict[str, Any]]:
    def retriever_node(state: ArchitectState) -> Dict[str, Any]:
        plan = state.get("retrieval_plan") or {}
        evidence: Dict[str, Any] = {"infrastructure": [], "packages": [], "models": [], "errors": [], "calls": []}
        seen: Dict[str, set] = {b: set() for b in BUCKETS.values()}

        for query in plan.get("queries", []) if plan.get("should_retrieve") else []:
            kb = query["kb"]
            bucket = BUCKETS.get(kb)
            tool = tools.get(kb)
            if bucket is None or tool is None:
                evidence["calls"].append({"kb": kb, "status": "skipped", "reason": "no tool for this knowledge base"})
                continue

            filters = query.get("filters") or {}
            if kb == "cloud_pricing_db":
                kwargs = {k: v for k, v in filters.items() if k in CLOUD_FILTER_KEYS and v is not None}
            elif kb == "k8s_marketplace":
                kwargs = {"user_query": query["query_text"], "top_k": query.get("top_k", 5)}
            else:
                kwargs = {"query": query["query_text"], "top_k": query.get("top_k", 5)}

            try:
                payload = tool(**kwargs) or {}
            except Exception as exc:  # a failing tool must not crash the graph
                payload = {"results": [], "error": str(exc)}

            rows: List[Dict[str, Any]] = payload.get("results") or []
            for row in rows:
                key = _row_key(bucket, row)
                if key not in seen[bucket]:
                    seen[bucket].add(key)
                    evidence[bucket].append(row)

            if payload.get("error"):
                evidence["errors"].append({"kb": kb, "error": payload["error"]})
            evidence["calls"].append(
                {"kb": kb, "intent": query.get("intent"), "args": kwargs, "rows": len(rows), "warnings": payload.get("warnings") or []}
            )

        return {"evidence": evidence}

    return retriever_node
