"""Agent and routing tests with a scripted LLM and stubbed tools.

No API key, database or LangGraph install is needed: the nodes are plain
functions, so the tests walk them in graph order using the real routing
functions.
"""

import copy
import json

import pytest

from ai_architect.agents import (
    make_clarifier_node,
    make_extractor_node,
    make_final_node,
    make_planner_node,
    make_retriever_node,
)
from ai_architect.llm_json import LLMJsonError, call_llm_json, extract_json
from ai_architect.routing import route_after_clarifier, route_after_planner
from ai_architect.state import empty_requirements, merge_requirements, missing_critical


class Reply:
    def __init__(self, content):
        self.content = content


class ScriptedLLM:
    """Returns a canned reply per agent, picked by that agent's system prompt."""

    MARKERS = {
        "extractor": "Requirements Extractor Agent",
        "clarifier": "Clarifying Agent",
        "planner": "Retrieval Planner Agent",
        "final": "Final Response Agent",
    }

    def __init__(self, **replies):
        self.replies = {k: list(v) if isinstance(v, list) else [v] for k, v in replies.items()}
        self.calls = []

    def invoke(self, messages):
        system = messages[0]["content"]
        agent = next(name for name, marker in self.MARKERS.items() if marker in system)
        self.calls.append(agent)
        queue = self.replies[agent]
        reply = queue.pop(0) if len(queue) > 1 else queue[0]
        return Reply(reply if isinstance(reply, str) else json.dumps(reply))


def full_requirements():
    req = empty_requirements()
    req["goal"].update(problem_domain="medical_imaging", task_type="image-segmentation")
    req["data"].update(modality="image", dimensionality="3D", dataset_size_tb=2)
    req["training"].update(strategy="fine_tune", framework="pytorch")
    req["deployment"].update(environment="cloud_k8s", compliance=["GDPR"])
    return req


PLAN = {
    "should_retrieve": True,
    "blocked_reason": None,
    "queries": [
        {"kb": "cloud_pricing_db", "intent": "gpu_selection", "query_text": "GPU for 3D segmentation fine-tuning",
         "filters": {"min_vram_gb": 48, "preference": "balanced", "region": "ignored"}, "top_k": 5},
        {"kb": "k8s_marketplace", "intent": "component_selection", "query_text": "pytorch distributed training operator", "top_k": 3},
        {"kb": "hf_models", "intent": "model_recommendation", "query_text": "3d medical image segmentation", "top_k": 3},
    ],
}

INFRA_ROW = {
    "provider": "ExampleCloud", "instance_type": "gpu-a100-x1", "gpu_model": "A100", "gpu_count": 1,
    "gpu_vram_total_gb": 80.0, "gpu_vram_per_gpu_gb": 80.0, "cpu_cores": 16, "ram_gb": 120.0,
    "price_per_hour_instance": 2.5, "stock_status": "available", "value_score": 100.0, "price_flag": "ok",
}


def make_tools(calls, infra=(INFRA_ROW,), packages=({"name": "kubeflow", "version": "1.0", "description": "ML workflows"},)):
    def cloud(**kwargs):
        calls.append(("cloud_pricing_db", kwargs))
        return {"results": list(infra)}

    def k8s(**kwargs):
        calls.append(("k8s_marketplace", kwargs))
        return {"results": list(packages)}

    def hf(**kwargs):
        calls.append(("hf_models", kwargs))
        return {"results": [{"model_id": "org/seg-model", "license": "apache-2.0"}]}

    return {"cloud_pricing_db": cloud, "k8s_marketplace": k8s, "hf_models": hf}


def final_draft():
    return {
        "ready_to_answer": True,
        "blocked_reason": None,
        "assumptions": ["Fine-tuning for about 100 GPU hours."],
        "gpu_recommendations": [
            {"label": "balanced", "provider": "examplecloud", "instance_type": "GPU-A100-X1", "gpu_type": "H100",
             "gpu_count": 8, "vram_gb": 999, "price_per_hour": 0.01, "why": "Fits 3D volumes in memory.",
             "fallbacks": ["made-up-instance"]},
            {"label": "invented", "provider": "NoSuchCloud", "instance_type": "fantasy-gpu", "gpu_type": "X",
             "gpu_count": 1, "vram_gb": 24, "price_per_hour": 1.0, "why": "This one is not in evidence.",
             "fallbacks": ["x"]},
        ],
        "model_recommendations": [
            {"model_id": "org/seg-model", "task": "image-segmentation", "license": "mit", "params_billion": 0.1,
             "min_vram_gb": 24, "notes": "Found by search."},
            {"model_id": "nnU-Net", "task": "image-segmentation", "license": "apache-2.0", "params_billion": 0.03,
             "min_vram_gb": 24, "notes": "Strong default."},
        ],
        "k8s_components": [
            {"category": "orchestration", "primary": "Kubeflow", "alternatives": ["airflow"], "why": "Pipelines for training."},
            {"category": "serving", "primary": "not-in-evidence", "alternatives": ["x"], "why": "Hallucinated component."},
        ],
        "architecture_diagram_mermaid": "flowchart TD; A[Storage]-->B[Training]-->C[Registry]-->D[Serving]",
        "deployment_plan": ["Provision cluster.", "Install Kubeflow.", "Run fine-tuning.", "Deploy model."],
        "cost_estimate": {"method": "$/hr x 100 hours x 1 GPU", "rough_range_usd": "$250 (derived)"},
    }


def run_turn(state, user_text, llm, tools):
    """Walk the nodes in graph order using the real routing functions."""
    state = copy.deepcopy(state)
    state.setdefault("messages", []).append({"role": "user", "content": user_text})

    def apply(update):
        update = dict(update)
        state["messages"].extend(update.pop("messages", []))
        state.update(update)

    apply(make_extractor_node(llm)(state))
    apply(make_clarifier_node(llm)(state))
    if route_after_clarifier(state) == "ask_user":
        return state
    apply(make_planner_node(llm)(state))
    if route_after_planner(state) == "retriever":
        apply(make_retriever_node(tools)(state))
    apply(make_final_node(llm)(state))
    return state


# ---------------------------------------------------------------- unit tests

def test_extract_json_handles_fences_and_prose():
    assert json.loads(extract_json('```json\n{"a": 1}\n```')) == {"a": 1}
    assert json.loads(extract_json('Sure! {"a": {"b": 2}} hope it helps')) == {"a": {"b": 2}}


def test_call_llm_json_repairs_then_gives_up():
    schema = {"type": "object", "required": ["x"], "properties": {"x": {"type": "integer"}}}

    class Flaky:
        def __init__(self, replies):
            self.replies, self.prompts = list(replies), []

        def invoke(self, messages):
            self.prompts.append(messages[1]["content"])
            return Reply(self.replies.pop(0))

    llm = Flaky(["not json", '{"x": 3}'])
    assert call_llm_json(llm, "sys", "in", schema) == {"x": 3}
    assert "VALIDATION ERROR" in llm.prompts[1]

    with pytest.raises(LLMJsonError):
        call_llm_json(Flaky(['{"x": "a"}'] * 3), "sys", "in", schema, max_retries=2)


def test_merge_never_downgrades_known_values_and_recomputes_gates():
    current = full_requirements()
    update = empty_requirements()  # everything unknown
    update["training"]["strategy"] = "inference_only"
    update["decision_gates"] = {"tool_ready": True, "missing_critical": ["bogus"]}
    update["extra"] = {"hallucinated": True}

    merged = merge_requirements(current, update)
    assert merged["training"]["strategy"] == "inference_only"   # real change applied
    assert merged["data"]["dataset_size_tb"] == 2                # not wiped by "unknown"
    assert "extra" not in merged
    assert merged["decision_gates"] == {"tool_ready": True, "missing_critical": []}


def test_substring_ct_no_longer_sets_modality():
    # The old regex code set modality=CT for any text containing "ct"
    # ("detect", "architecture"). Extraction is now the LLM's job, and an
    # all-unknown reply must leave the requirements unknown.
    llm = ScriptedLLM(extractor=empty_requirements())
    out = make_extractor_node(llm)({"messages": [{"role": "user", "content": "detect objects, need architecture"}]})
    assert out["requirements"]["data"]["modality"] == "unknown"
    assert set(out["requirements"]["decision_gates"]["missing_critical"]) == set(missing_critical(empty_requirements()))


# ------------------------------------------------------------ pipeline tests

def test_two_turn_flow_clarify_then_grounded_answer():
    partial = full_requirements()
    partial["training"]["strategy"] = "unknown"
    question = {"field": "training.strategy", "question": "Train from scratch, fine-tune or inference only?",
                "why_it_matters": "Decides GPU memory tier.", "options": ["train_from_scratch", "fine_tune", "inference_only"]}

    llm = ScriptedLLM(
        extractor=[partial, full_requirements()],
        clarifier=[{"questions": [question]}, {"questions": []}],
        planner=PLAN,
        final=final_draft(),
    )
    calls = []
    tools = make_tools(calls)

    # Turn 1: one critical field missing -> ask, and do NOT touch the tools.
    state = run_turn({}, "2TB of MRI volumes, segment tumours, PyTorch on cloud k8s, GDPR", llm, tools)
    assert [q["field"] for q in state["pending_questions"]] == ["training.strategy"]
    assert calls == [] and "planner" not in llm.calls
    assert state["clarify_rounds"] == 1

    # Turn 2: answered -> plan -> retrieve -> grounded final.
    state = run_turn(state, "fine-tune", llm, tools)
    assert state["pending_questions"] == []
    assert llm.calls == ["extractor", "clarifier", "extractor", "planner", "final"]

    # The planner's arguments reached the tools; unknown filter keys were dropped.
    assert calls[0] == ("cloud_pricing_db", {"min_vram_gb": 48, "preference": "balanced"})
    assert calls[1] == ("k8s_marketplace", {"user_query": "pytorch distributed training operator", "top_k": 3})
    assert calls[2][0] == "hf_models"

    final = state["final_response"]
    assert final["ready_to_answer"] is True

    # Hallucinated GPU dropped; surviving one carries evidence values, not the LLM's.
    assert len(final["gpu_recommendations"]) == 1
    gpu = final["gpu_recommendations"][0]
    assert (gpu["provider"], gpu["instance_type"]) == ("ExampleCloud", "gpu-a100-x1")
    assert (gpu["gpu_type"], gpu["gpu_count"], gpu["vram_gb"], gpu["price_per_hour"]) == ("A100", 1, 80.0, 2.5)
    assert gpu["fallbacks"] == ["no alternative in retrieved evidence"]

    # Hallucinated k8s component dropped; alternatives restricted to evidence.
    assert [c["primary"] for c in final["k8s_components"]] == ["kubeflow"]
    assert final["k8s_components"][0]["alternatives"] == ["none in retrieved evidence"]

    # Verified model keeps evidence licence; unverified one is labelled.
    verified, baseline = final["model_recommendations"]
    assert verified["license"] == "apache-2.0"
    assert baseline["license"] == "unknown" and baseline["params_billion"] is None
    assert "not tool-verified" in baseline["notes"]


def test_blocks_without_llm_when_tools_return_nothing():
    llm = ScriptedLLM(extractor=full_requirements(), clarifier={"questions": []}, planner=PLAN, final=final_draft())
    state = run_turn({}, "everything specified", llm, make_tools([], infra=()))

    final = state["final_response"]
    assert final["ready_to_answer"] is False
    assert "GPU/cloud pricing" in final["blocked_reason"]
    assert "final" not in llm.calls  # no evidence -> the final LLM is never asked to guess


def test_blocks_when_every_recommendation_is_hallucinated():
    draft = final_draft()
    draft["gpu_recommendations"] = draft["gpu_recommendations"][1:]  # only the invented one
    llm = ScriptedLLM(extractor=full_requirements(), clarifier={"questions": []}, planner=PLAN, final=draft)
    state = run_turn({}, "everything specified", llm, make_tools([]))
    assert state["final_response"]["ready_to_answer"] is False
    assert state["final_response"]["gpu_recommendations"] == []


def test_question_budget_then_blocked_report(monkeypatch):
    import ai_architect.agents.clarifier as clarifier_module

    monkeypatch.setattr(clarifier_module, "MAX_CLARIFY_ROUNDS", 1)
    llm = ScriptedLLM(extractor=empty_requirements(), clarifier="garbage that is not json")
    calls = []

    # Round 1: model output unusable -> deterministic fallback questions.
    state = run_turn({}, "help me with AI", llm, make_tools(calls))
    assert len(state["pending_questions"]) == 3

    # Round 2: budget spent and still unknown -> blocked, no tools, no planner LLM.
    state = run_turn(state, "I don't know", llm, make_tools(calls))
    assert state["final_response"]["ready_to_answer"] is False
    assert "Missing critical requirements" in state["final_response"]["blocked_reason"]
    assert calls == [] and "planner" not in llm.calls
