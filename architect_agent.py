

import sys

# Core

# Your libs

import os
import re
import numpy as np
import json
import difflib
from jsonschema import validate
import psycopg2
import psycopg2.extras
from psycopg2 import OperationalError
from pathlib import Path
from typing import Annotated, List, TypedDict, Optional, Dict, Any
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from huggingface_hub import HfApi, list_models
from collections import Counter

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mistral")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-latest")





llm = ChatOpenAI(
     model="gpt-4o",
     temperature=0,
     api_key="sk-proj-Z9C"
 )




def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# Load schemas
REQ_SCHEMA = load_json("Schema/requirements.json")
CLARIFY_SCHEMA = load_json("Schema/clarifying.json")
PLAN_SCHEMA = load_json("Schema/retrieval_plan.json")
FINAL_SCHEMA = load_json("Schema/final_response.json")

# Load prompts
REQ_SYSTEM = load_text("prompts/requirements.txt")
CLARIFY_SYSTEM = load_text("prompts/clarifying.txt")
PLAN_SYSTEM = load_text("prompts/retrieval_plan.txt")
FINAL_SYSTEM = load_text("prompts/final_response.txt")



#SEARCH_MODE_TEMPLATE
SEARCH_MODE_TEMPLATE = """
YOU ARE IN SEARCH MODE. YOUR ONLY TASK IS TO GATHER FACTS USING TOOLS.

RULES:
1. DO NOT OUTPUT ANY TEXT, JSON, OR EXPLANATIONS
2. DO NOT MAKE RECOMMENDATIONS
3. DO NOT ANSWER THE USER'S QUESTION
4. YOUR ONLY OUTPUT SHOULD BE TOOL CALLS
5. CALL MULTIPLE TOOLS TO GATHER COMPREHENSIVE DATA

AVAILABLE TOOLS:
1. search_cloud_infrastructure - For GPU instances, pricing, availability
2. search_huggingface_models - For AI models, licenses, VRAM requirements
3. search_packages - For software packages, Kubernetes components

BASED ON THESE REQUIREMENTS, CALL ALL RELEVANT TOOLS:

{requirements_json}

OUTPUT ONLY TOOL CALLS. NO OTHER TEXT.
"""

# Tools

# Global, loaded once
EMBEDDING_MODEL = SentenceTransformer("all-mpnet-base-v2")

DB_URL = "postgresql://postgres:oR9yZ4i.ai:5432/stack8s-cloud"

CACHE_WHITELIST = [
    r"\bcache\b", r"\bcaching\b", r"\bredis\b", r"\bmemcached\b",
    r"\bin-memory\b", r"\bkey-?value\b", r"\bsentinel\b", r"\bcluster\b"
]

CACHE_BLACKLIST = [
    r"\bmysql\b", r"\bpostgres\b", r"\bpostgresql\b",
    r"\bmongodb\b", r"\bmariadb\b", r"\bphpmyadmin\b"
]

# Compile regex once (faster + cleaner)
_CACHE_WHITELIST_RE = [re.compile(p, re.IGNORECASE) for p in CACHE_WHITELIST]
_CACHE_BLACKLIST_RE = [re.compile(p, re.IGNORECASE) for p in CACHE_BLACKLIST]

def _is_cache_related(text: str) -> bool:
    t = text or ""
    if any(rx.search(t) for rx in _CACHE_BLACKLIST_RE):
        return False
    return any(rx.search(t) for rx in _CACHE_WHITELIST_RE)


def _normalize_query(user_query: str) -> Dict[str, Any]:
    q = (user_query or "").strip()
    t = q.lower()
    is_cache_intent = any(k in t for k in ["cache", "caching", "in-memory", "redis", "memcached"])
    norm = "redis cache" if is_cache_intent else q
    return {"query": q, "normalized": norm, "is_cache_intent": is_cache_intent}


@tool
def search_packages(user_query: str, top_k: int = 5) -> dict:
    """
    Semantic search over cloud.bitnami_packages using embeddings.
    Returns evidence-only JSON results (no recommendations).
    """

    # ---- input normalization ----
    nq = _normalize_query(user_query)
    q = nq["query"]
    norm = nq["normalized"]
    is_cache_intent = nq["is_cache_intent"]

    warnings: List[str] = []

    if not q:
        return {"source": "bitnami_packages", "results": [], "warnings": ["empty query"]}

    try:
        top_k = int(top_k or 5)
        top_k = max(1, min(top_k, 25))  # safety cap
    except Exception:
        top_k = 5
        warnings.append("Invalid top_k; defaulted to 5.")

    # ---- embedding ----
    try:
        query_embedding = EMBEDDING_MODEL.encode(norm).tolist()
    except Exception as e:
        return {
            "source": "bitnami_packages",
            "query": q,
            "normalized_query": norm,
            "results": [],
            "error": f"Embedding error: {str(e)}"
        }

    conn = None
    try:
        conn = psycopg2.connect(DB_URL, connect_timeout=10)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Pull more than top_k, then filter + trim
        limit = max(30, top_k * 10)

        cursor.execute(
            """
            WITH ranked AS (
              SELECT
                name, description, category, stars, version,
                (description_embedding <-> %s::vector) AS desc_dist
              FROM cloud.bitnami_packages
              WHERE description_embedding IS NOT NULL
            ),
            dedup AS (
              SELECT DISTINCT ON (name)
                name, description, category, stars, version, desc_dist
              FROM ranked
              ORDER BY name, desc_dist ASC, stars DESC
            )
            SELECT *
            FROM dedup
            ORDER BY desc_dist ASC, stars DESC
            LIMIT %s;
            """,
            (query_embedding, limit)
        )

        rows = cursor.fetchall() or []
        cursor.close()

        if not rows:
            return {
                "source": "bitnami_packages",
                "query": q,
                "normalized_query": norm,
                "results": [],
                "warnings": warnings + ["No rows returned."]
            }

        filtered_rows = rows

        if is_cache_intent:
            filtered_rows = [
                r for r in rows
                if _is_cache_related(f"{r.get('name','')} {r.get('description','')}")
            ]
            if not filtered_rows:
                warnings.append(
                    "No strongly cache-related packages found; returning closest semantic matches instead."
                )
                filtered_rows = rows

        filtered_rows = filtered_rows[:top_k]

        return {
            "source": "bitnami_packages",
            "query": q,
            "normalized_query": norm,
            "results": [
                {
                    "name": r.get("name"),
                    "category_id": r.get("category"),
                    "description": r.get("description"),
                    "stars": r.get("stars"),
                    "version": r.get("version"),
                    "desc_dist": float(r.get("desc_dist")) if r.get("desc_dist") is not None else None
                }
                for r in filtered_rows
            ],
            "warnings": warnings
        }

    except Exception as e:
        return {"source": "bitnami_packages", "query": q, "normalized_query": norm, "results": [], "error": str(e)}
    finally:
        if conn:
            conn.close()

def is_suspicious_price(row):
    provider = (row.get("provider") or "").lower()
    gpu_txt = f"{row.get('gpu_model','')} {row.get('name','')} {row.get('instance_type','')}".lower()
    price = float(row.get("price_hourly") or row.get("price_per_hour_instance") or 0)
    return provider == "aws" and ("h100" in gpu_txt or "a100" in gpu_txt) and price > 0 and price < 2.0


def _derive_vram(gpu_count: int, gpu_memory_gb: float):
    """
    Heuristic:
    - If gpu_memory_gb <= 96: treat as PER-GPU VRAM, total = per * count
    - Else: treat as TOTAL VRAM, per = total / count
    """
    if gpu_count <= 0:
        return None, float(gpu_memory_gb or 0)

    gm = float(gpu_memory_gb or 0)
    if gm <= 96:
        per_gpu = gm
        total = gm * gpu_count
    else:
        total = gm
        per_gpu = (gm / gpu_count) if gpu_count else None

    return per_gpu, total


@tool
def search_cloud_infrastructure(
    min_cpu: int = 0,
    min_ram_gb: int = 0,
    min_vram_gb: int = 0,  # interpreted as MIN TOTAL VRAM per instance (after derivation)
    provider: str = None,
    preference: str = "balanced"
):
    """
    Search GPU cloud instances from cloud.instances with basic filters and ranking.
    Returns evidence-only JSON: instance price/hour, per-GPU price, VRAM totals, stock status.
    """
    warnings = []

    preference = (preference or "balanced").strip().lower()
    if preference not in ("cheapest", "performance", "balanced"):
        warnings.append(f"Invalid preference '{preference}' received; defaulting to 'balanced'.")
        preference = "balanced"

    try:
        min_cpu = int(min_cpu or 0)
        min_ram_gb = int(min_ram_gb or 0)
        min_vram_gb = int(min_vram_gb or 0)
    except Exception:
        warnings.append("Invalid numeric constraints received; defaulting mins to 0.")
        min_cpu, min_ram_gb, min_vram_gb = 0, 0, 0

    provider = (provider or "").strip() or None

    conn = None
    try:
        conn = psycopg2.connect(DB_URL, connect_timeout=10)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        query = """
            SELECT
              provider,
              name,
              instance_type,
              cpu_cores,
              memory_gb,
              gpu_model,
              gpu_count,
              gpu_memory_gb,
              price_hourly,
              price_currency,
              stock_status
            FROM cloud.instances
            WHERE available = True
              AND gpu_count IS NOT NULL AND gpu_count > 0
              AND gpu_memory_gb IS NOT NULL AND gpu_memory_gb > 0
              AND price_hourly IS NOT NULL AND price_hourly > 0
        """
        params = []

        if min_cpu > 0:
            query += " AND cpu_cores >= %s"
            params.append(min_cpu)
        if min_ram_gb > 0:
            query += " AND memory_gb >= %s"
            params.append(min_ram_gb)
        if provider:
            query += " AND provider ILIKE %s"
            params.append(f"%{provider}%")

        limit = 200 if preference == "performance" else 80
        query += " ORDER BY price_hourly ASC LIMIT %s"
        params.append(limit)

        cursor.execute(query, tuple(params))
        results = cursor.fetchall()
        cursor.close()

    except OperationalError:
        return {"source": "cloud_pricing_db", "warnings": warnings, "results": [], "error": "Connection Error: Cannot reach the database."}
    except Exception as e:
        return {"source": "cloud_pricing_db", "warnings": warnings, "results": [], "error": f"Unexpected Error: {str(e)}"}
    finally:
        if conn:
            conn.close()

    if not results:
        return {"source": "cloud_pricing_db", "warnings": warnings, "results": [], "error": "No instances found matching your criteria."}

    enriched = []
    for r in results:
        gpu_count = int(r.get("gpu_count") or 0)
        gm = float(r.get("gpu_memory_gb") or 0)
        per_gpu, total = _derive_vram(gpu_count, gm)

        # apply min_vram_gb as TOTAL VRAM
        if min_vram_gb > 0 and total < min_vram_gb:
            continue

        price_hour = float(r.get("price_hourly") or 0)
        cpu = float(r.get("cpu_cores") or 0)
        ram = float(r.get("memory_gb") or 0)

        perf = (total * 3.0) + (cpu * 1.0) + (ram * 0.5)
        value = perf / (price_hour if price_hour > 0 else 1e9)

        r["_vram_total"] = total
        r["_vram_per_gpu"] = per_gpu
        r["_performance_score"] = perf
        r["_value_score"] = value

        enriched.append(r)

    if not enriched:
        warnings.append("No instances matched derived VRAM constraints after VRAM derivation.")
        return {"source": "cloud_pricing_db", "warnings": warnings, "results": []}

    # Dedup (provider, instance_type, gpu_model, vram_total)
    unique = {}
    for r in enriched:
        key = (r.get("provider"), r.get("instance_type") or r.get("name"), r.get("gpu_model"), round(float(r["_vram_total"]), 3))
        price = float(r.get("price_hourly") or 1e9)
        if key not in unique or price < float(unique[key].get("price_hourly") or 1e9):
            unique[key] = r

    deduped = list(unique.values())

    if preference == "cheapest":
        sorted_results = sorted(deduped, key=lambda x: float(x.get("price_hourly") or 1e9))
    elif preference == "performance":
        sorted_results = sorted(deduped, key=lambda x: (-x["_performance_score"], float(x.get("price_hourly") or 1e9)))
    else:
        sorted_results = sorted(deduped, key=lambda x: (is_suspicious_price(x), -x["_value_score"], float(x.get("price_hourly") or 1e9)))

    out = []
    for r in sorted_results[:5]:
        gpu_count = int(r.get("gpu_count") or 0)
        price_hour = float(r.get("price_hourly") or 0)

        total = float(r["_vram_total"])
        per_gpu = r["_vram_per_gpu"]
        price_per_gpu_hour = (price_hour / gpu_count) if gpu_count > 0 else None

        out.append({
            "provider": r.get("provider"),
            "instance_type": r.get("instance_type") or r.get("name"),
            "gpu_model": r.get("gpu_model"),
            "gpu_count": gpu_count,
            "gpu_vram_total_gb": round(total, 2),
            "gpu_vram_per_gpu_gb": round(float(per_gpu), 2) if per_gpu is not None else None,
            "cpu_cores": int(r.get("cpu_cores") or 0),
            "ram_gb": round(float(r.get("memory_gb") or 0), 2),
            "price_per_hour_instance": round(price_hour, 6),
            "price_currency": r.get("price_currency"),
            "price_per_gpu_hour": round(price_per_gpu_hour, 6) if price_per_gpu_hour is not None else None,
            "stock_status": r.get("stock_status"),
            "value_score": round(float(r["_value_score"] or 0.0), 2),
            "price_flag": "suspicious" if is_suspicious_price(r) else "ok"
        })

    warnings.append("VRAM derivation is heuristic: gpu_memory_gb may be per-GPU or total depending on provider feed.")
    return {"source": "cloud_pricing_db", "warnings": warnings, "results": out}



from huggingface_hub import HfApi
from typing import Dict, Any, List
import re, math

api = HfApi()

# File extensions that usually contain model weights
WEIGHT_EXTS = (
    ".safetensors", ".bin", ".pth", ".pt", ".ckpt", ".onnx",
    ".h5", ".tflite", ".mlmodel", ".pdparams", ".npz", ".msgpack"
)

# "Transformers-style" important files (common for NLP/ASR/LLMs, sometimes CV)
IMPORTANT_FILES = (
    "config.json",
    "tokenizer.json", "tokenizer_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "generation_config.json",
)

STOP = {
    "the","a","an","and","or","with","for","to","of","in","on",
    "model","models","best","top","hf","huggingface"
}

# -------------------------
# Text utilities
# -------------------------
def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    toks = re.findall(r"[a-z0-9][a-z0-9_\-]+", text)
    return [t for t in toks if t not in STOP and len(t) > 1]

def overlap(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a))

# -------------------------
# Repo file evidence (heavy → run only for top candidates)
# -------------------------
def repo_file_evidence(repo_id: str) -> Dict[str, Any]:
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    except Exception as e:
        return {
            "files_checked": False,
            "error": str(e),
            "files_count": 0,
            "has_weights": False,
            "important_files": [],
            "weight_files_sample": [],
            "adapter_files_sample": [],
        }

    lower = [f.lower() for f in files]
    weight_files = [f for f in files if f.lower().endswith(WEIGHT_EXTS)]
    important_found = [f for f in IMPORTANT_FILES if f.lower() in lower]
    adapter_files = [f for f in files if "adapter" in f.lower()]

    return {
        "files_checked": True,
        "files_count": len(files),
        "has_weights": len(weight_files) > 0,
        "important_files": important_found,
        "weight_files_sample": weight_files[:10],
        "adapter_files_sample": adapter_files[:10],
    }

def get_license(info) -> str:
    card = getattr(info, "cardData", None) or getattr(info, "card_data", None)
    if isinstance(card, dict) and card.get("license"):
        return str(card["license"])
    for t in getattr(info, "tags", []) or []:
        if isinstance(t, str) and t.startswith("license:"):
            return t.split("license:", 1)[1] or "unknown"
    return "unknown"

# -------------------------
# Scoring (GENERAL)
# -------------------------
def relevance_score(query: str, model_id: str, tags: List[str], task: str, library: str) -> float:
    q = set(tokenize(query))
    if not q:
        return 0.0

    mid = set(tokenize(model_id))
    tagt = set(tokenize(" ".join(tags or [])))
    meta = set(tokenize(" ".join([task or "", library or ""])))

    s_id   = overlap(q, mid)
    s_tag  = overlap(q, tagt)
    s_meta = overlap(q, meta)

    return 2.2*s_id + 1.4*s_tag + 0.6*s_meta

def task_bias_from_query(query: str, task: str) -> float:
    """
    VERY LIGHT intent hints from query → prefer matching pipeline_tag.
    Still general (works for many domains).
    """
    q = (query or "").lower()
    t = (task or "unknown").lower()

    bias = 0.0

    # Segmentation intent
    if any(k in q for k in ("segmentation", "segment", "mask", "contour")):
        if t == "image-segmentation":
            bias += 2.5
        elif t != "unknown":
            bias -= 0.8
        else:
            bias -= 1.5

    # Classification intent
    if any(k in q for k in ("classification", "classify", "categorization")):
        if t == "image-classification":
            bias += 2.0

    # ASR intent
    if any(k in q for k in ("asr", "speech recognition", "speech-to-text", "transcription")):
        if t == "automatic-speech-recognition":
            bias += 2.0

    # Text generation / LLM-ish intent
    if any(k in q for k in ("llm", "chat", "instruct", "instruction", "text generation", "text-generation")):
        if t in ("text-generation", "text2text-generation"):
            bias += 1.8

    return bias

def trust_score(task: str, library: str, license_: str, evidence: Dict[str, Any]) -> float:
    score = 0.0

    if task and task != "unknown":
        score += 1.5
    if library and library != "unknown":
        score += 1.0
    if license_ and license_ != "unknown":
        score += 2.0

    if evidence.get("has_weights"):
        score += 3.0

    imp = evidence.get("important_files") or []

    # "Transformers-style" files help, but not required for all CV repos.
    # So we only give a modest bonus.
    if len(imp) >= 1:
        score += 0.8
    if len(imp) >= 3:
        score += 0.6

    # If we couldn't check files, reduce confidence a bit
    if evidence.get("files_checked") is False:
        score -= 0.8

    return score

def popularity_score(downloads: int, likes: int) -> float:
    downloads = int(downloads or 0)
    likes = int(likes or 0)
    return math.log10(downloads + 1) + 0.35 * math.log10(likes + 1)

def final_score(query: str, r: Dict[str, Any]) -> float:
    rel = relevance_score(query, r["model_id"], r.get("tags") or [], r.get("task","unknown"), r.get("library","unknown"))
    tru = trust_score(r.get("task","unknown"), r.get("library","unknown"), r.get("license","unknown"), r.get("evidence") or {})
    pop = popularity_score(r.get("downloads",0), r.get("likes",0))
    bias = task_bias_from_query(query, r.get("task","unknown"))

    # Weights: relevance > trust > popularity, plus task bias
    return (rel * 4.0) + (tru * 2.5) + (pop * 1.0) + (bias * 3.0)

# -------------------------
# Main search (GENERAL + relevance-aware + efficient)
# -------------------------
def search_huggingface_models(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    General-purpose Hugging Face model search.
    Returns top_k models ranked by: relevance + trust + popularity,
    with a light task-bias derived from the query (segmentation/ASR/LLM/etc).

    Efficiency:
    - Pull many candidates (cheap).
    - Fetch model_info for top N candidates.
    - Fetch repo files evidence ONLY for top M candidates (heavy).
    """
    query = (query or "").strip()
    if not query:
        return {"source": "hf_models", "results": [], "warnings": ["empty query"]}

    try:
        # 1) Retrieve many candidates (cheap)
        candidate_pool = max(800, top_k * 80)
        raw_candidates = list(api.list_models(
            search=query,
            sort="downloads",
            direction=-1,
            limit=candidate_pool
        ))

        # Take more than 50 so niche "segmentation" repos not named "lung cancer" can appear
        candidates = raw_candidates[:150]

        temp: List[Dict[str, Any]] = []

        # 2) Light metadata (moderate cost)
        for it in candidates:
            try:
                info = api.model_info(it.id)
            except Exception:
                continue

            model_id = getattr(info, "modelId", it.id)
            task = getattr(info, "pipeline_tag", None) or "unknown"
            library = getattr(info, "library_name", None) or "unknown"
            tags = getattr(info, "tags", None) or []

            downloads = int(getattr(info, "downloads", 0) or 0)
            likes = int(getattr(info, "likes", 0) or 0)

            temp.append({
                "model_id": model_id,
                "task": task,
                "library": library,
                "license": get_license(info),
                "downloads": downloads,
                "likes": likes,
                "tags": tags[:40],
                # placeholder evidence; filled for top candidates later
                "evidence": {
                    "files_checked": False,
                    "files_count": 0,
                    "has_weights": False,
                    "important_files": [],
                    "weight_files_sample": [],
                    "adapter_files_sample": [],
                },
            })

        if not temp:
            return {"source": "hf_models", "results": [], "warnings": ["No models returned from HF API"]}

        # 3) First rerank (without repo files evidence)
        temp.sort(key=lambda r: final_score(query, r), reverse=True)

        # 4) Heavy repo file checks only for top candidates
        heavy_check_n = max(top_k * 3, 20)  # e.g. top 20
        for r in temp[:heavy_check_n]:
            r["evidence"] = repo_file_evidence(r["model_id"])

        # 5) Final rerank with evidence included
        temp.sort(key=lambda r: final_score(query, r), reverse=True)

        return {"source": "hf_models", "results": temp[:top_k], "warnings": []}

    except Exception as e:
        return {"source": "hf_models", "results": [], "warnings": [str(e)]}

template = """

You are the expert AI Systems Architect for stack8.

Your goal is to move from a user idea to a concrete, deployable AI infrastructure plan,
while providing expert-level, trustworthy recommendations for AI models and infrastructure.

────────────────────────────────────────
OPERATING MODES — YOU MUST FOLLOW EXACTLY ONE MODE AT A TIME
────────────────────────────────────────

CURRENT CONTEXT AND EXTRACTED REQUIREMENTS ARE INJECTED ABOVE.
USE THEM TO DECIDE WHICH MODE TO ENTER.

If CURRENT CONTEXT or REQUIREMENTS are missing or incomplete,
you MUST default to CLARIFICATION MODE.

────────────────────────────────────────
MODEL DISCOVERY & FALLBACK POLICY
────────────────────────────────────────

- When a model discovery tool is available, you SHOULD attempt to use it.
- If the model discovery tool returns:
  - no results,
  - empty or null results,
  - models unrelated to the workload,
  - low-quality or generic repositories,
  - or results lacking medical relevance or usable weights,

  then the results are considered INVALID.

- In case of INVALID model results:
  - You MUST fall back to expert domain reasoning.
  - You MUST recommend a suitable Hugging Face reference model
    based on common research practice and domain knowledge.
  - Such recommendations MUST be framed as research or educational baselines.
  - You MUST NOT claim real-time verification or clinical approval.

────────────────────────────────────────
1. CLARIFICATION MODE
────────────────────────────────────────

Enter this mode ONLY if:
- The domain/workload is clear (e.g., vision, NLP, medical imaging)
- BUT critical parameters are missing that affect GPU sizing or architecture choice

Rules:
- Output ONLY valid JSON matching the ClarifyingQuestions schema
- Ask at most 3 focused questions
- Questions must directly impact:
  - VRAM requirements
  - GPU count
  - Multi-GPU strategy
  - Training vs fine-tuning vs inference
- NEVER mention specific GPU models
- NEVER call tools
- NEVER output final recommendations or prose

────────────────────────────────────────
2. SEARCH MODE
────────────────────────────────────────

Enter this mode when:
- All necessary sizing parameters are known
- No further clarification is required

CRITICAL RULES FOR SEARCH MODE:
- Your ONLY job is to gather factual infrastructure data using tools
- You MAY call one or more relevant infrastructure tools
- You MUST NOT write any text, explanation, summary, or recommendation
- You MUST NOT manually output JSON
- You MUST NOT hallucinate prices, availability, regions, or compliance constraints

Available tools:
- search_cloud_infrastructure(query: str)
- search_packages(query: str)

Output:
- ONLY tool calls
- NOTHING else

────────────────────────────────────────
3. FINAL RESPONSE MODE
────────────────────────────────────────

Enter this mode ONLY when:
- All required infrastructure tool results are available in the conversation history
- No pending clarification exists

ROLE IN THIS MODE:
You are an AI Architect specialized in medical imaging and large-scale ML systems.

Rules:
- Output ONLY valid JSON matching the AIArchitectFinalResponse schema
- Hugging Face model recommendations MUST be generated as follows:
  - Use valid model discovery tool results IF available
  - Otherwise, fall back to expert domain reasoning per the fallback policy
- Do NOT search Hugging Face or claim real-time verification
- Recommended models must be framed as research/educational baselines
- Accuracy metrics must be described as reported or typical, not clinically guaranteed
- GPU, cloud, pricing, and availability MUST be grounded in tool outputs
- NEVER call tools
- NEVER ask questions

Final response MUST include:
1. Short restatement of the user problem
2. Recommended Hugging Face reference model(s) with explanation
3. Model architecture, training background, and typical metrics
4. How to load and fine-tune the model
5. GPU recommendation (from tool results)
6. Cloud & compliance recommendation (from tool results)
7. Clear disclaimer: research use only, not clinical
8. Optional alternative models
9. Explicit assumptions
10. Confidence level of the recommendation

────────────────────────────────────────
MODE DECISION CHECKLIST
────────────────────────────────────────

- If ANY of the following are unknown:
  training strategy,
  deployment environment,
  data scale,
  compliance constraints
  → CLARIFICATION MODE

- Else if infrastructure tools have NOT been called
  → SEARCH MODE

- Else
  → FINAL RESPONSE MODE

────────────────────────────────────────
GLOBAL RULES — NEVER VIOLATE
────────────────────────────────────────

- Never mix modes
- Never output raw text outside the allowed schema or tool calls
- Never claim clinical or regulatory approval unless explicitly stated
- Never assume sensitive constraints without evidence
- Prioritize accuracy, traceability, and safety over speed

NOW:
Decide which mode to enter and follow its rules EXACTLY.


"""

import json, re
from jsonschema import validate
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Any, Dict, List, Optional


def _extract_json(text: str) -> str:
    text = text.strip()

    # 1) Prefer fenced blocks
    fence = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()

    # 2) Try to find first valid JSON object/array by scanning
    # (simple balanced scan; avoids grabbing unrelated braces)
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1].strip()
                    # quick sanity parse
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        break

    return text  # fallback (may fail json.loads)


def _maybe_unwrap(data: Any, schema: dict) -> Any:
    title = (schema or {}).get("title", "")

    if not isinstance(data, dict):
        return data

    # Requirements wrapper
    if title == "AIArchitectRequirements" and "requirements" in data and isinstance(data["requirements"], dict):
        return data["requirements"]

    # Final wrapper (common mistake: {"recommendation": {...}})
    if title == "AIArchitectFinalResponse" and "recommendation" in data and isinstance(data["recommendation"], dict):
        return data["recommendation"]

    return data


def sanitize_clarify(data: Any) -> dict:
    if not isinstance(data, dict) or "questions" not in data:
        return {"questions": []}

    sanitized = []
    for q in data["questions"]:
        if not isinstance(q, dict):
            continue

        expected = q.get("expected_answer_type", "free_text")

        item = {
            "category": q.get("category", "deployment_preference"),
            "question": str(q.get("question", "")).strip(),
            "why_it_matters": str(q.get("why_it_matters", "This impacts infrastructure planning.")).strip(),
            "expected_answer_type": expected
        }

        # ONLY include options if multiple_choice and has valid options
        if expected == "multiple_choice":
            opts = q.get("options", [])
            if isinstance(opts, list):
                opts = [str(o).strip() for o in opts if str(o).strip()]
            if len(opts) >= 2:
                item["options"] = opts

        if item["question"]:  # only add if there's a real question
            sanitized.append(item)

    return {"questions": sanitized[:5]}  # max 5 questions


def sanitize_final(data: Any) -> dict:
    """Sanitize final response JSON - remove any extra fields"""
    if not isinstance(data, dict):
        return {}

    # Define expected fields from FINAL_SCHEMA
    expected_fields = {
        "assumptions", "gpu_recommendations", "model_recommendations",
        "k8s_components", "architecture_diagram_mermaid",
        "deployment_plan", "cost_estimate"
    }

    # Remove any fields not in the schema
    keys_to_remove = [k for k in data.keys() if k not in expected_fields]
    for k in keys_to_remove:
        del data[k]

    return data


def sanitize_requirements(data: Any) -> dict:
    """Remove any extra fields not in REQ_SCHEMA from requirements"""
    if not isinstance(data, dict):
        return {}

    # Define expected fields from requirements schema
    expected_fields = {
        "user_goal", "dataset", "problem", "preferences", "inferred_tech_stack"
    }

    # Remove any fields not in the schema
    keys_to_remove = [k for k in data.keys() if k not in expected_fields]
    for k in keys_to_remove:
        del data[k]

    # Ensure all required fields exist with proper defaults
    for field in expected_fields:
        if field not in data:
            if field == "user_goal":
                data[field] = ""
            elif field == "dataset":
                data[field] = {
                    "size": {"value": None, "unit": "unknown", "raw_description": "unknown"},
                    "data_type": "unknown",
                    "modality": "unknown",
                    "format": "unknown",
                    "dimensionality": "unknown",
                    "labeling": "unknown"
                }
            elif field == "problem":
                data[field] = {
                    "domain": "unknown",
                    "task": "unknown",
                    "target": "unknown",
                    "constraints": {"latency": "unknown", "throughput": "unknown", "accuracy_priority": "unknown"},
                    "ambiguities": []
                }
            elif field == "preferences":
                data[field] = {
                    "framework": "unknown",
                    "deployment_env": "unknown",
                    "budget_strategy": "unknown"
                }
            elif field == "inferred_tech_stack":
                data[field] = {
                    "model_architecture_family": ["Other"],
                    "training_strategy": "unknown",
                    "compute_profile": {"needs_gpu": False, "vram_requirement": "unknown", "multi_gpu_scaling": "unknown"},
                    "supporting_tools": []
                }

    return data



def call_llm_json_mistral(llm, system_prompt: str, user_content, schema: dict, max_retries: int = 2) -> dict:
    original_user_content = user_content
    if not isinstance(user_content, str):
        user_content = json.dumps(user_content, ensure_ascii=False)

    strict_suffix = (
        "\n\nCRITICAL OUTPUT RULES:\n"
        "- Output ONLY valid JSON. No markdown, no code blocks, no wrappers.\n"
        "- Do not add any keys not defined in the schema.\n"
        "- Do not explain or comment.\n"
    )

    last_text = None

    for attempt in range(max_retries + 1):
        messages = [
            SystemMessage(content=system_prompt + strict_suffix),
            HumanMessage(content=user_content)
        ]

        response = llm.invoke(messages)
        text = (response.content or "").strip()
        last_text = text

        json_text = _extract_json(text)

        try:
            data = json.loads(json_text)
            data = _maybe_unwrap(data, schema)

            title = (schema or {}).get("title", "")

            # Apply schema-specific sanitization
            if title == "AIArchitectRequirements":
                data = sanitize_requirements(data)
            elif title in {"ClarificationAgent", "Clarification", "ClarifyingQuestions"}:
                data = sanitize_clarify(data)
            elif title in {"AIArchitectFinalResponse", "AI Architect Final Response"}:
                data = sanitize_final(data)

            # Validate
            validate(instance=data, schema=schema)

        except Exception as e:
            if attempt < max_retries:
                user_content = (
                    "Your JSON did not match the schema.\n"
                    "Fix it to match EXACTLY. Output JSON only.\n\n"
                    f"VALIDATION ERROR:\n{str(e)[:1200]}\n\n"
                    f"USER REQUEST:\n{original_user_content if isinstance(original_user_content, str) else json.dumps(original_user_content, ensure_ascii=False)}"
                )
                continue
            raise ValueError(f"JSON does not match schema:\n{e}\n\nJSON:\n{data}")

        return data

    raise ValueError(f"LLM did not return valid JSON after retries.\nLast output:\n{last_text}")




# ----------------------------
# Intent/Task Inference (LLM -> requirements patch)
# ----------------------------
INTENT_SCHEMA = {
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "IntentExtraction",
  "type": "object",
  "additionalProperties": False,
  "required": ["problem_domain", "task_type", "modality", "dimensionality", "dataset_size_tb", "confidence", "rationale"],
  "properties": {
    "problem_domain": {
      "type": "string",
      "enum": ["medical_imaging", "computer_vision", "nlp", "speech_audio", "tabular", "time_series", "recommender", "multimodal", "unknown"]
    },
    "task_type": {
      "type": "string",
      "enum": [
        "image-segmentation", "object-detection", "image-classification",
        "text-classification", "token-classification", "question-answering", "text-generation",
        "automatic-speech-recognition", "audio-classification",
        "tabular-classification", "tabular-regression",
        "unknown"
      ]
    },
    "modality": { "type": "string", "enum": ["image", "text", "audio", "tabular", "multimodal", "unknown"] },
    "dimensionality": { "type": "string", "enum": ["3D", "2D", "1D", "unknown"] },
    "dataset_size_tb": { "type": ["number", "null"], "minimum": 0 },
    "confidence": { "type": "number", "minimum": 0, "maximum": 1 },
    "rationale": { "type": "string", "minLength": 10 }
  }
}

INTENT_SYSTEM_PROMPT = (
    "You are an intent extraction module for an AI Architect agent. "
    "Given a user's free-text request, infer the MOST LIKELY ML problem domain and Hugging Face task type. "
    "If it is ambiguous (e.g., 'detect cancer' could be classification vs detection vs segmentation), "
    "pick the best default but LOWER confidence. "
    "Output STRICT JSON that matches the provided schema."
)

def infer_intent_patch(llm, user_text: str) -> dict:
    """Infer domain/task/modality and return a small patch dict for requirements."""
    if not user_text or not isinstance(user_text, str):
        return {}

    result = call_llm_json_mistral(
        llm,
        system_prompt=INTENT_SYSTEM_PROMPT,
        user_content=user_text,
        schema=INTENT_SCHEMA,
        max_retries=2,
    )

    # Minimal patch to avoid breaking existing flat requirements shape
    patch = {
        "problem_domain": result.get("problem_domain", "unknown"),
        "task_type": result.get("task_type", "unknown"),
        "data_modality": result.get("modality", "unknown"),
        "data_dimensionality": result.get("dimensionality", "unknown"),
        "dataset_size_tb": result.get("dataset_size_tb", None),
        "intent_confidence": result.get("confidence", None),
        "intent_rationale": result.get("rationale", ""),
    }
    return patch

def augment_model_query(model_query: str, requirements: dict) -> str:
    """Add domain/task hints to the HF query string without duplicating too much."""
    q = (model_query or "").strip()
    task = (requirements or {}).get("task_type")
    domain = (requirements or {}).get("problem_domain")
    modality = (requirements or {}).get("data_modality")
    dim = (requirements or {}).get("data_dimensionality")

    task_phrase_map = {
        "image-segmentation": "segmentation",
        "object-detection": "object detection",
        "image-classification": "image classification",
        "text-classification": "text classification",
        "token-classification": "token classification",
        "question-answering": "question answering",
        "text-generation": "text generation",
        "automatic-speech-recognition": "speech recognition",
        "audio-classification": "audio classification",
        "tabular-classification": "tabular classification",
        "tabular-regression": "tabular regression",
    }

    hints = []
    if domain and domain != "unknown":
        hints.append(domain.replace("_", " "))
    if task and task != "unknown":
        hints.append(task_phrase_map.get(task, task.replace("-", " ")))
    if modality and modality != "unknown":
        hints.append(modality)
    if dim and dim != "unknown":
        hints.append(dim)

    hint_str = " ".join(dict.fromkeys(hints))  # de-dupe preserving order
    if not hint_str:
        return q or "foundation model fine-tuning"

    # If the user already wrote an explicit task, don't spam hints.
    q_low = (q or "").lower()
    if task and task != "unknown" and task.replace("-", " ") in q_low:
        return q or hint_str

    return f"{hint_str} {q}".strip()

tools = [search_cloud_infrastructure, search_packages]


tool_llm = llm.bind_tools(tools)

# Optional schema validation
try:
    from jsonschema import validate as js_validate
    from jsonschema.exceptions import ValidationError
except Exception:
    js_validate = None
    ValidationError = Exception


# ----------------------------
# State
# ----------------------------
class ArchitectState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    current_question: Optional[Dict[str, Any]]
    answers: Dict[str, str]
    requirements: Optional[Dict[str, Any]]
    phase: str
    tool_evidence: List[Dict[str, Any]]
    expected_tool_names: List[str]
    search_issued: bool


# ----------------------------
# Helpers (schema safety + QA normalization)
# ----------------------------

def sanitize_model_recommendations(items: Any) -> List[Dict[str, Any]]:
    """
    Enforce schema: model_recommendations items MUST contain ONLY:
    model_name, params_billion, license, notes
    """
    out: List[Dict[str, Any]] = []
    if not isinstance(items, list):
        return out

    for x in items:
        if not isinstance(x, dict):
            continue

        out.append({
            "model_name": str(x.get("model_name") or "").strip(),
            "params_billion": x.get("params_billion", None),
            "license": str(x.get("license") or "unknown").strip() or "unknown",
            "notes": str(x.get("notes") or "").strip(),
        })

    # drop empty model_name rows
    return [m for m in out if m["model_name"]]



def _normalize_to_schema(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Force known-problem fields into schema-safe shapes."""
    # deployment_plan -> list[str]
    dp = parsed.get("deployment_plan")
    fixed_dp: List[str] = []
    if isinstance(dp, list):
        for item in dp:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    fixed_dp.append(s)
            elif isinstance(item, dict):
                step = item.get("step")
                action = (item.get("action") or "").strip()
                details = (item.get("details") or "").strip()
                prefix = f"Step {step}: " if step is not None else ""
                s = (prefix + action).strip()
                if details:
                    s = (s + " " + details).strip()
                if s:
                    fixed_dp.append(s)
    parsed["deployment_plan"] = fixed_dp

    # assumptions -> list[str]
    a = parsed.get("assumptions")
    if isinstance(a, list):
        parsed["assumptions"] = [str(x) for x in a]
    else:
        parsed["assumptions"] = []

    return parsed


def _q_key(q: Dict[str, Any]) -> str:
    return str(q.get("category") or q.get("question") or "")


def normalize_choice(text: str) -> str:
    t = (text or "").strip().lower()
    mapping = {
        "no": "none", "none": "none", "n/a": "none",
        "doesn't matter": "none", "doesnt matter": "none",
        "not important": "none", "irrelevant": "none", "non": "none",

        "aws": "cloud_k8s", "amazon": "cloud_k8s",
        "amazon web services": "cloud_k8s", "cloud": "cloud_k8s",
        "kubernetes": "cloud_k8s", "k8s": "cloud_k8s",
        "cloud_k8s": "cloud_k8s",

        "fine-tuning": "fine_tune", "fine tuning": "fine_tune",
        "finetuning": "fine_tune", "fine tuned": "fine_tune",
        "fine-tune": "fine_tune", "fine tune": "fine_tune",

        "train from scratch": "train_from_scratch",
        "from scratch": "train_from_scratch",

        "inference": "inference_only", "inference only": "inference_only",

        "pytorch": "pytorch", "torch": "pytorch", "py torch": "pytorch",
        "tensorflow": "tensorflow", "tf": "tensorflow", "tensor flow": "tensorflow",

        "no preference": "no_preference", "no_preference": "no_preference",
        "either": "no_preference",
    }
    return mapping.get(t, (text or "").strip())


def normalize_answer_for_question(q: Dict[str, Any], user_text: str) -> str:
    t = (user_text or "").strip()

    if q.get("expected_answer_type") != "multiple_choice":
        return normalize_choice(t)

    options = q.get("options") or []
    if not options:
        return normalize_choice(t)

    t2 = normalize_choice(t).lower()
    opts_lower = [o.lower() for o in options]
    match = difflib.get_close_matches(t2, opts_lower, n=1, cutoff=0.6)
    if match:
        return options[opts_lower.index(match[0])]

    return normalize_choice(t)


def _json_question_msg(question_dict: Dict[str, Any]) -> AIMessage:
    content = json.dumps({"questions": [question_dict]}, indent=2, ensure_ascii=False)
    return AIMessage(content=content)


# ----------------------------
# Base Questions
# ----------------------------
training_strategy_question = {
    "category": "training_strategy",
    "question": "Will you be training from scratch, fine-tuning a pre-trained model, or doing inference only?",
    "why_it_matters": "This determines the GPU VRAM tier, GPU count, and expected training time/cost.",
    "expected_answer_type": "multiple_choice",
    "options": ["train_from_scratch", "fine_tune", "inference_only"],
}

compliance_question = {
    "category": "compliance_governance",
    "question": "Are there any compliance, legal, or governance constraints (e.g., GDPR, HIPAA, data residency, or internal policies)?",
    "why_it_matters": "This affects where the compute can run (cloud/on-prem), what storage/encryption is required, and audit needs.",
    "expected_answer_type": "free_text",
}

deployment_question = {
    "category": "deployment_preference",
    "question": "Where do you want this to run: cloud Kubernetes, on-prem, hybrid, edge, or air-gapped?",
    "why_it_matters": "This determines what infra options are feasible and what GPU instances/vendors we can recommend.",
    "expected_answer_type": "multiple_choice",
    "options": ["cloud_k8s", "on_prem", "hybrid", "edge_device", "air_gapped"],
}

framework_question = {
    "category": "framework_preference",
    "question": "Do you have a framework preference for the ML pipeline (PyTorch, TensorFlow), or no preference?",
    "why_it_matters": "This affects model/library choices, training stack, serving options, and GPU utilization patterns.",
    "expected_answer_type": "multiple_choice",
    "options": ["pytorch", "tensorflow", "no_preference"],
}


def build_pending_questions(requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
    pending: List[Dict[str, Any]] = []
    if requirements.get("training_strategy") is None:
        pending.append(training_strategy_question)
    if requirements.get("compliance_governance") is None:
        pending.append(compliance_question)
    if requirements.get("deployment_preference") is None:
        pending.append(deployment_question)
    if requirements.get("framework_preference") is None:
        pending.append(framework_question)
    return pending



def extract_implicit_requirements(
    user_text: str,
    requirements: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    requirements = requirements or {}
    raw_text = user_text or ""
    text = raw_text.lower().strip()

    # ----------------------------
    # 1) Always store the raw query (intent)
    # ----------------------------
    if requirements.get("model_query") is None:
        requirements["model_query"] = raw_text.strip()

    # ----------------------------
    # 2) Training strategy
    # ----------------------------
    if requirements.get("training_strategy") is None:
        if re.search(r"\bfine[\s-]?tun(e|ing|ed)?\b", text):
            requirements["training_strategy"] = "fine_tune"
        elif "from scratch" in text or "train from scratch" in text:
            requirements["training_strategy"] = "train_from_scratch"
        elif "inference" in text:
            requirements["training_strategy"] = "inference_only"

    # ----------------------------
    # 3) Framework preference
    # ----------------------------
    if requirements.get("framework_preference") is None:
        if "pytorch" in text or re.search(r"\btorch\b", text):
            requirements["framework_preference"] = "pytorch"
        elif "tensorflow" in text or re.search(r"\btf\b", text):
            requirements["framework_preference"] = "tensorflow"

    # ----------------------------
    # 4) Deployment preference
    # ----------------------------
    if requirements.get("deployment_preference") is None:
        if "k8s" in text or "kubernetes" in text or "cloud" in text:
            requirements["deployment_preference"] = "cloud_k8s"
        elif "on prem" in text or "on-prem" in text or "onprem" in text:
            requirements["deployment_preference"] = "on_prem"
        elif "air gap" in text or "air-gapped" in text:
            requirements["deployment_preference"] = "air_gapped"
        elif "edge" in text:
            requirements["deployment_preference"] = "edge_device"

    # ----------------------------
    # 5) ML task (use ml_task, don't create "task")
    # ----------------------------
    if requirements.get("ml_task") is None:
        if "segment" in text or "segmentation" in text or "unet" in text:
            requirements["ml_task"] = "segmentation"
        elif "detect" in text or "detection" in text:
            requirements["ml_task"] = "detection"
        elif "classif" in text or "classification" in text:
            requirements["ml_task"] = "classification"

    # ----------------------------
    # 6) Dataset extraction (ONLY if explicit)
    # ----------------------------
    if requirements.get("dataset_size") is None:
        m = re.search(r"(\d+(?:\.\d+)?)\s*(tb|gb)\b", text)
        if m:
            requirements["dataset_size"] = {
                "value": float(m.group(1)),
                "unit": m.group(2).lower()
            }

    if requirements.get("modality") is None:
        if "mri" in text:
            requirements["modality"] = "MRI"
        elif "ct" in text:
            requirements["modality"] = "CT"
        elif "x-ray" in text or "xray" in text:
            requirements["modality"] = "X-ray"

    return requirements



# ----------------------------
# Final Prompt + Schema
# ----------------------------
FINAL_PROMPT = """
You are the AI Architect – Final Response Agent.

Return ONLY a single valid JSON object that conforms EXACTLY to the AIArchitectFinalResponse schema.

ABSOLUTE OUTPUT RULES
- Output pure JSON only (no markdown, no backticks, no explanations).
- Do NOT add any keys not defined in the schema.
- All required fields MUST be present.

EVIDENCE RULES
- GPU types, instance types, prices, availability, and Kubernetes component choices MUST be grounded strictly in tool evidence.
- If pricing/availability evidence is missing, use null where allowed and explain in assumptions.

MODEL FALLBACK RULES
- Model recommendations MAY be generated from expert domain reasoning if model tool evidence is missing.
- Do NOT claim real-time verification of model existence, availability, or metrics.
- Do NOT claim clinical/regulatory approval.
- If model evidence is missing: set license="unknown", params_billion=null,
  and notes must say "expert baseline / estimate (not tool-verified)".

IMPORTANT
- deployment_plan MUST be an array of strings only (each item a single sentence). Never output objects for deployment_plan.
- Output ONLY the JSON object.
"""

AIARCH_FINAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "ready_to_answer",
        "blocked_reason",
        "assumptions",
        "gpu_recommendations",
        "model_recommendations",
        "k8s_components",
        "architecture_diagram_mermaid",
        "deployment_plan",
        "cost_estimate"
    ],
    "properties": {
        "ready_to_answer": {"type": "boolean"},
        "blocked_reason": {"type": ["string", "null"]},
        "assumptions": {"type": "array", "items": {"type": "string"}},

        "gpu_recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "provider", "instance_type",
                    "gpu_model", "gpu_vram_gb",
                    "cpu_cores", "ram_gb",
                    "price_per_hour", "stock_status",
                    "value_score"
                ],
                "properties": {
                    "provider": {"type": "string"},
                    "instance_type": {"type": "string"},
                    "gpu_model": {"type": "string"},
                    "gpu_vram_gb": {"type": "integer", "minimum": 0},
                    "cpu_cores": {"type": "integer", "minimum": 0},
                    "ram_gb": {"type": "integer", "minimum": 0},
                    "price_per_hour": {"type": "number", "minimum": 0},
                    "stock_status": {"type": ["string", "null"]},
                    "value_score": {"type": "number"}
                }
            }
        },

        "model_recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["model_name", "params_billion", "license", "notes"],
                "properties": {
                    "model_name": {"type": "string"},
                    "params_billion": {"type": ["number", "null"]},
                    "license": {"type": "string"},
                    "notes": {"type": "string"}
                }
            }
        },

        "k8s_components": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "version", "description"],
                "properties": {
                    "name": {"type": "string"},
                    "version": {"type": ["string", "null"]},
                    "description": {"type": ["string", "null"]}
                }
            }
        },

        "architecture_diagram_mermaid": {"type": "string"},
        "deployment_plan": {"type": "array", "items": {"type": "string"}},

        "cost_estimate": {
            "type": "object",
            "additionalProperties": False,
            "required": ["method", "rough_range_usd"],
            "properties": {
                "method": {"type": "string"},
                "rough_range_usd": {"type": "string"}
            }
        }
    }
}


def _safe_blocked_json(reason: str, extra_assumptions=None) -> Dict[str, Any]:
    extra_assumptions = extra_assumptions or []
    return {
        "ready_to_answer": False,
        "blocked_reason": reason or "Final response could not be parsed/validated as JSON.",
        "assumptions": ["Final response JSON could not be parsed/validated from the LLM output."] + extra_assumptions,
        "gpu_recommendations": [],
        "model_recommendations": [],
        "k8s_components": [],
        "architecture_diagram_mermaid": "",
        "deployment_plan": [],
        "cost_estimate": {
            "method": "No reliable infrastructure/pricing JSON could be produced due to invalid final output.",
            "rough_range_usd": "not available"
        }
    }


def _extract_json_substring(text: str) -> str:
    if not text:
        return ""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start:end + 1].strip()


def _try_parse_json(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("Empty output")
    return json.loads(text)


def adapt_llm_final_to_schema(d: dict) -> dict:
    """Convert common LLM output shape into AIARCH_FINAL_SCHEMA shape."""
    if not isinstance(d, dict):
        return {}

    out = {
        "ready_to_answer": True,
        "blocked_reason": None,
        "assumptions": [],
        "gpu_recommendations": [],
        "model_recommendations": [],
        "k8s_components": [],
        "architecture_diagram_mermaid": "",
        "deployment_plan": [],
        "cost_estimate": {"method": "tool-evidence + estimates", "rough_range_usd": "not available"},
    }

    # assumptions
    if isinstance(d.get("assumptions"), list):
        out["assumptions"] = [str(x) for x in d["assumptions"] if str(x).strip()]

    # model_recommendations
    mr = d.get("model_recommendations")
    if isinstance(mr, list):
        out["model_recommendations"] = sanitize_model_recommendations(mr)
    elif isinstance(d.get("model_recommendation"), dict):
        out["model_recommendations"] = sanitize_model_recommendations([d["model_recommendation"]])


    # gpu_recommendations pass-through (will be overwritten by evidence later)
    gr = d.get("gpu_recommendations")
    if isinstance(gr, list):
        out["gpu_recommendations"] = gr

    # k8s_components pass-through (will be overwritten by evidence later)
    kc = d.get("k8s_components")
    if isinstance(kc, list):
        out["k8s_components"] = kc

    # deployment_plan
    dp = d.get("deployment_plan")
    if isinstance(dp, list):
        out["deployment_plan"] = [str(x).strip() for x in dp if str(x).strip()]

    # mermaid diagram
    if isinstance(d.get("architecture_diagram_mermaid"), str):
        out["architecture_diagram_mermaid"] = d["architecture_diagram_mermaid"]

    # cost_estimate (condense)
    ce = d.get("cost_estimate")
    if isinstance(ce, dict):
        out["cost_estimate"] = {
            "method": "Derived from tool evidence; detailed estimate condensed.",
            "rough_range_usd": ce.get("rough_range_usd") or "see assumptions"
        }

    # ready/blocked pass-through if present (but you override later anyway)
    if isinstance(d.get("ready_to_answer"), bool):
        out["ready_to_answer"] = d["ready_to_answer"]
    if "blocked_reason" in d:
        out["blocked_reason"] = d.get("blocked_reason")

    return out



def _get_price(row: dict) -> float:
    """
    Pick the best available hourly price from tool evidence.
    Handles old + new tool output shapes.
    """
    for k in ["price_per_hour", "price_hourly", "price_per_hour_instance", "price_per_gpu_hour"]:
        v = row.get(k)
        if v is None:
            continue
        try:
            v = float(v)
            if v > 0:
                return v
        except Exception:
            continue
    return 0.0


def _get_vram_per_gpu(row: dict):
    """
    Prefer per-GPU VRAM if available; else derive from total/gpu_count.
    """
    v = row.get("gpu_vram_per_gpu_gb")
    if v is not None:
        try:
            return int(round(float(v)))
        except Exception:
            pass

    total = row.get("gpu_vram_total_gb")
    if total is not None:
        try:
            total = float(total)
            gc = int(row.get("gpu_count") or 0)
            if gc > 0:
                return int(round(total / gc))
            return int(round(total))
        except Exception:
            pass

    # legacy fields
    v2 = row.get("gpu_vram_gb")
    if v2 is not None:
        try:
            return int(round(float(v2)))
        except Exception:
            pass

    return 0


def build_gpu_recommendations_from_evidence(evidence: dict, top_k: int = 3) -> list:
    rows = (evidence or {}).get("infrastructure") or []
    out = []

    def _f(v, default=0.0):
        try:
            return float(v)
        except Exception:
            return float(default)

    def _i(v, default=0):
        try:
            return int(float(v))
        except Exception:
            return int(default)

    def _pick_price(r: dict) -> float:
        for k in ["price_per_hour", "price_per_hour_instance", "price_hourly", "price_per_hourly"]:
            v = r.get(k)
            if v is not None and _f(v) > 0:
                return _f(v)
        return 0.0

    def _pick_vram_total(r: dict) -> float:
        for k in ["gpu_vram_total_gb", "gpu_vram_gb", "gpu_memory_gb"]:
            v = r.get(k)
            if v is not None and _f(v) > 0:
                return _f(v)
        return 0.0

    def _pick_gpu_count(r: dict) -> int:
        return _i(r.get("gpu_count") or 0)

    def _pick_ram_gb(r: dict) -> int:
        for k in ["ram_gb", "memory_gb"]:
            v = r.get(k)
            if v is not None and _f(v) > 0:
                return _i(v)
        return 0

    def _pick_instance_type(r: dict) -> str:
        return (r.get("instance_type") or r.get("name") or r.get("instance_id") or "").strip()

    for r in rows[:top_k]:
        gpu_count = _pick_gpu_count(r)
        vram_total = _pick_vram_total(r)
        vram_per_gpu = (vram_total / gpu_count) if gpu_count > 0 else vram_total

        out.append({
            "provider": (r.get("provider") or "").strip(),
            "instance_type": _pick_instance_type(r),
            "gpu_model": (r.get("gpu_model") or "").strip(),
            # schema فقط یک عدد می‌خواد؛ ما per-GPU می‌دیم چون برای تصمیم بهتره
            "gpu_vram_gb": int(round(vram_per_gpu)),
            "cpu_cores": _i(r.get("cpu_cores") or 0),
            "ram_gb": _pick_ram_gb(r),
            "price_per_hour": float(_pick_price(r)),
            "stock_status": r.get("stock_status"),
            "value_score": float(r.get("value_score") or 0.0),
        })

    return out




def build_k8s_components_from_evidence(evidence: dict, top_k: int = 5) -> list:
    pkgs = (evidence or {}).get("packages") or []
    out = []
    for p in pkgs[:top_k]:
        out.append({
            "name": str(p.get("name") or ""),
            "version": p.get("version"),
            "description": p.get("description"),
        })
    return out


def build_deployment_plan_grounded(requirements: dict, gpu_recs: list, k8s_components: list) -> list:
    top = gpu_recs[0] if gpu_recs else {}
    inst = (top.get("instance_type") or "GPU nodes").strip()
    prov = (top.get("provider") or "cloud").strip()
    fw = (requirements.get("framework_preference") or "tensorflow").strip()

    # Only use dataset_description if explicitly available; otherwise keep generic.
    dataset_desc = requirements.get("dataset_description")
    dataset_phrase = f"the dataset ({dataset_desc})" if dataset_desc else "the dataset"

    comp_names = {c.get("name") for c in (k8s_components or []) if c.get("name")}
    has_kubeflow = "kubeflow" in comp_names
    has_training_op = "training-operator" in comp_names
    has_gfd = "gpu-feature-discovery" in comp_names
    has_seldon = "seldon" in comp_names

    plan = []
    plan.append(f"Provision a Kubernetes cluster on {prov} with GPU-enabled nodes matching the selected instance type ({inst}).")
    if has_kubeflow:
        plan.append("Install Kubeflow on the cluster for ML workflow orchestration.")
    if has_gfd:
        plan.append("Install gpu-feature-discovery to label GPU nodes and enable correct scheduling.")
    if has_training_op:
        plan.append(f"Install training-operator and run distributed {fw} fine-tuning jobs as Kubernetes custom resources.")
    plan.append(f"Store {dataset_phrase} in encrypted storage and mount it into training jobs with least-privilege access.")
    plan.append("Add GDPR controls: EU-region residency, encryption at rest/in transit, audit logging, and retention policies.")
    plan.append("Validate the fine-tuned model on a held-out set and log metrics for reproducibility.")
    plan.append("Deploy the validated model behind an inference service, with monitoring and access controls." if not has_seldon
                else "Deploy the validated model using Seldon for inference, with monitoring and access controls.")
    return plan


# ----------------------------
# Nodes
# ----------------------------
def architect_node(state: ArchitectState):
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else None

    current_q = state.get("current_question")
    answers = state.get("answers") or {}
    requirements = state.get("requirements") or {}
    phase = state.get("phase") or "qa"
    tool_evidence = state.get("tool_evidence") or []
    expected_tool_names = state.get("expected_tool_names") or []
    search_issued = state.get("search_issued") or False

    print(
        f"DEBUG architect_node: phase={phase}, current_q={current_q}, "
        f"last_msg_type={type(last_msg).__name__ if last_msg else None}"
    )

    # 1) Collect ONLY NEW ToolMessages (tail block)
    tool_msgs = []
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            tool_msgs.append(m)
        else:
            break
    tool_msgs = list(reversed(tool_msgs))

    if tool_msgs:
        for tm in tool_msgs:
            try:
                payload = tm.content
                if isinstance(payload, str):
                    payload = json.loads(payload)
            except Exception:
                payload = {"raw": tm.content}

            tool_evidence.append({"name": getattr(tm, "name", None), "payload": payload})

        got_names = {e.get("name") for e in tool_evidence if e.get("name")}
        expected = set(expected_tool_names)
        done = bool(expected) and expected.issubset(got_names)

        return {
            "messages": [],
            "phase": "done" if done else "qa",  # go back to QA if tools missing
            "tool_evidence": tool_evidence,
            "expected_tool_names": expected_tool_names,
            "search_issued": search_issued,
            "requirements": requirements,
            "answers": answers,
            "current_question": None,
        }

    # 2) Implicit extraction
    if isinstance(last_msg, HumanMessage):
        requirements = extract_implicit_requirements(last_msg.content, requirements)

    # 3) Q&A answer handling
    if current_q is not None and isinstance(last_msg, HumanMessage):
        key = _q_key(current_q)
        normalized = normalize_answer_for_question(current_q, last_msg.content)
        answers[key] = normalized

        cat = current_q.get("category")
        if cat in {"training_strategy", "deployment_preference", "compliance_governance", "framework_preference"}:
            requirements[cat] = normalized

        pending = build_pending_questions(requirements)
        if pending:
            next_q = pending[0]
            return {
                "messages": [_json_question_msg(next_q)],
                "requirements": requirements,
                "current_question": next_q,
                "answers": answers,
                "phase": "qa",
                "tool_evidence": tool_evidence,
                "expected_tool_names": expected_tool_names,
                "search_issued": search_issued,
            }

        # 4) Base questions done -> search once
        if not search_issued:
            print("DEBUG SEARCH requirements snapshot:")
            print(json.dumps(requirements, indent=2, ensure_ascii=False))

            tool_calls = [
                {
                    "type": "tool_call",
                    "name": "search_cloud_infrastructure",
                    "args": {"min_vram_gb": 24, "preference": "performance"},
                    "id": "cloud_gpu_search",
                },
                {
                    "type": "tool_call",
                    "name": "search_packages",
                    "args": {
                        "user_query": (
                            "Kubernetes packages for GPU training and serving of "
                            "medical imaging models (PyTorch, TensorFlow, MONAI, nnUNet)"
                        ),
                        "top_k": 5,
                    },
                    "id": "k8s_packages_search",
                },
            ]

            response = AIMessage(content="", tool_calls=tool_calls)
            expected_tool_names = sorted(list({tc["name"] for tc in tool_calls}))

            return {
                "messages": [response],
                "requirements": requirements,
                "current_question": None,
                "answers": answers,
                "phase": "search",
                "tool_evidence": tool_evidence,
                "expected_tool_names": expected_tool_names,
                "search_issued": True,
            }

    # 5) Ask base questions (initial QA)
    if current_q is None and phase == "qa":
        pending = build_pending_questions(requirements)
        if pending:
            first_q = pending[0]
            return {
                "messages": [_json_question_msg(first_q)],
                "requirements": requirements,
                "current_question": first_q,
                "answers": answers,
                "phase": "qa",
                "tool_evidence": tool_evidence,
                "expected_tool_names": expected_tool_names,
                "search_issued": search_issued,
            }

    # 6) Default: no-op
    return {
        "messages": [],
        "requirements": requirements,
        "current_question": current_q,
        "answers": answers,
        "phase": phase,
        "tool_evidence": tool_evidence,
        "expected_tool_names": expected_tool_names,
        "search_issued": search_issued,
    }


def should_continue(state: ArchitectState):
    if state.get("phase") == "done":
        return "final"

    messages = state.get("messages", [])
    phase = state.get("phase") or "qa"
    current_q = state.get("current_question")

    print(f"DEBUG should_continue: phase={phase}, current_question={current_q}, messages={len(messages)}")

    # waiting for user answer
    if current_q is not None:
        if messages and isinstance(messages[-1], HumanMessage):
            return "architect"
        return END

    # IMPORTANT: do NOT end the graph just because search phase has no messages
    # (let architect node decide what to do next)
    if phase == "search" and not messages:
        return "architect"

    if not messages:
        return "architect"

    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        print(f"TOOL CALL DETECTED: {last_message.tool_calls}")
        return "action"

    if isinstance(last_message, ToolMessage):
        return "architect"

    return "architect"


def final_node(state: ArchitectState):
    print("DEBUG final_node: generating final JSON answer")
    requirements = state.get("requirements") or {}
    evidence_items = state.get("tool_evidence") or []

    def _get_results(payload: dict):
        for k in ["results", "items", "data", "output", "result"]:
            v = payload.get(k)
            if isinstance(v, list):
                return v
        return []

    evidence = {"infrastructure": [], "packages": []}
    for item in evidence_items:
        name = item.get("name")
        payload = item.get("payload") or {}
        results = _get_results(payload)
        if name == "search_cloud_infrastructure":
            evidence["infrastructure"].extend(results)
        elif name == "search_packages":
            evidence["packages"].extend(results)

    user_payload = {"requirements": requirements, "tool_evidence": evidence}

    # ---- 1) First attempt ----
    msg = llm.invoke([
        {"role": "system", "content": FINAL_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)}
    ])

    raw = getattr(msg, "content", None) or str(msg)
    raw = (raw or "").strip()

    parsed = None
    parse_error = None

    # ---- parse try (raw) ----
    for attempt in range(2):
        try:
            candidate = raw if attempt == 0 else _extract_json_substring(raw)
            parsed = _try_parse_json(candidate)
            break
        except Exception as e:
            parse_error = str(e)

    # ---- 2) Repair attempt ----
    if parsed is None:
        repair_prompt = f"""
    Return ONLY a single valid JSON object that conforms EXACTLY to this schema:
    {json.dumps(AIARCH_FINAL_SCHEMA, ensure_ascii=False)}

    Rules:
    - Output pure JSON only (no markdown, no explanations).
    - deployment_plan MUST be an array of strings.
    """
        msg2 = llm.invoke([
            {"role": "system", "content": repair_prompt},
            {"role": "user", "content": raw}
        ])
        raw2 = getattr(msg2, "content", None) or str(msg2)
        raw2 = (raw2 or "").strip()

        for attempt in range(2):
            try:
                candidate = raw2 if attempt == 0 else _extract_json_substring(raw2)
                parsed = _try_parse_json(candidate)
                break
            except Exception as e:
                parse_error = str(e)

    # ---- Adapt + normalize (only if dict) ----
    if not isinstance(parsed, dict):
        parsed = None
    else:
        parsed = adapt_llm_final_to_schema(parsed)
        parsed = _normalize_to_schema(parsed)
        parsed["model_recommendations"] = sanitize_model_recommendations(parsed.get("model_recommendations"))


        # Expert fallback model (NOT tool-verified)
        if not parsed.get("model_recommendations"):
            parsed["model_recommendations"] = [{
                "model_name": "nnU-Net (expert baseline for 3D medical imaging)",
                "params_billion": None,
                "license": "unknown",
                "notes": "expert baseline / estimate (not tool-verified); strong default for 3D segmentation."
            }]

        # ---- Deterministic grounding (truth from tools) ----
        parsed["gpu_recommendations"] = build_gpu_recommendations_from_evidence(evidence, top_k=3)
        parsed["k8s_components"] = build_k8s_components_from_evidence(evidence, top_k=5)

        # Ground deployment_plan from evidence (avoid hallucinations)
        parsed["deployment_plan"] = build_deployment_plan_grounded(
            requirements,
            parsed["gpu_recommendations"],
            parsed["k8s_components"]
        )

        # Grounded/minimal assumptions (avoid hallucinations)
        parsed["assumptions"] = [
            f"training_strategy={requirements.get('training_strategy')}",
            f"deployment_preference={requirements.get('deployment_preference')}",
            f"framework_preference={requirements.get('framework_preference')}",
            f"compliance_governance={requirements.get('compliance_governance')}",
            "GPU prices and instance attributes are taken directly from the cloud_pricing_db tool output.",
            "gpu_vram_gb is reported as per-GPU VRAM when gpu_count is available; otherwise it falls back to total VRAM from the DB row."
        ]

        # ---- Quality gate #1: tools returned nothing ----
        if not parsed.get("gpu_recommendations") or not parsed.get("k8s_components"):
            parsed["ready_to_answer"] = False
            parsed["blocked_reason"] = "Tool evidence missing for GPU or Kubernetes components."
        else:
            # ---- Quality gate #2: pricing sanity ----
            bad_prices = []
            for i, g in enumerate(parsed["gpu_recommendations"]):
                try:
                    p = float(g.get("price_per_hour") or 0.0)
                except Exception:
                    p = 0.0
                if p <= 0:
                    bad_prices.append(i)

            if bad_prices:
                parsed["ready_to_answer"] = False
                parsed["blocked_reason"] = "GPU pricing evidence has missing/zero hourly prices after normalization."
                parsed["assumptions"].append(
                    "At least one GPU recommendation had price_per_hour <= 0; likely field-mapping mismatch or DB has null/zero price."
                )
            else:
                parsed["ready_to_answer"] = True
                parsed["blocked_reason"] = None


    # ---- 3) Validate (optional) ----
    if parsed is not None and js_validate is not None:
        try:
            js_validate(instance=parsed, schema=AIARCH_FINAL_SCHEMA)
        except ValidationError as ve:
            parsed = None
            parse_error = f"Schema validation failed: {ve}"

    # ---- 4) Hard fallback ----
    if parsed is None:
        blocked = _safe_blocked_json(
            reason="Final response could not be parsed/validated as JSON.",
            extra_assumptions=[parse_error] if parse_error else []
        )
        return {
            "messages": [AIMessage(content=json.dumps(blocked, ensure_ascii=False))],
            "requirements": requirements
        }

    return {
        "messages": [AIMessage(content=json.dumps(parsed, ensure_ascii=False))],
        "requirements": requirements
    }


# ----------------------------
# Tools
# ----------------------------
# NOTE: you must define tools = [...] elsewhere
tool_node = ToolNode(tools)


# ----------------------------
# Graph
# ----------------------------
workflow = StateGraph(ArchitectState)
workflow.add_node("architect", architect_node)
workflow.add_node("action", tool_node)
workflow.add_node("final", final_node)

workflow.set_entry_point("architect")

workflow.add_conditional_edges(
    "architect",
    should_continue,
    {
        "action": "action",
        "final": "final",
        "architect": "architect",
        END: END
    }
)

workflow.add_edge("action", "architect")
workflow.add_edge("final", END)

memory = MemorySaver()
agent = workflow.compile(checkpointer=memory)

import json
from langchain_core.messages import HumanMessage, AIMessage

import json
import asyncio
from langchain_core.messages import HumanMessage, AIMessage

def extract_ai_text(event):
    # event structure usually like: {"architect": {"messages": [AIMessage(...)]}}
    for _, payload in event.items():
        msgs = payload.get("messages", [])
        for m in msgs:
            if isinstance(m, AIMessage) and m.content:
                return m.content
    return None

def _run_one_turn_sync(user_text: str, thread_id: str) -> dict:
    """
    Runs the (sync) LangGraph stream in a worker thread to avoid blocking Chainlit's event loop.
    """
    config = {"configurable": {"thread_id": thread_id}}

    final_text = None
    debug_obj = None
    events = []

    for event in agent.stream({"messages": [HumanMessage(content=user_text)]}, config):
        events.append(event)
        ai_text = extract_ai_text(event)
        if ai_text:
            final_text = ai_text
            try:
                debug_obj = json.loads(ai_text)
            except Exception:
                debug_obj = None

    return {
        "final_answer": final_text or "",
        "debug": debug_obj,
        # "events": events,  # optional: uncomment if you want heavy debug
    }

async def run_agent(user_text: str, history: list[dict], thread_id: str) -> dict:
    """
    Entry point used by orchestrator.py / Chainlit.
    Must accept (user_text, history, thread_id).
    """
    return await asyncio.to_thread(_run_one_turn_sync, user_text, thread_id)
















