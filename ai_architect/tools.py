"""Retrieval tools: cloud GPU pricing, Kubernetes packages, Hugging Face models.

Each tool returns evidence only (no recommendations) as
{"source": ..., "results": [...], "warnings": [...], "error": ...}.
Heavy dependencies are imported lazily so the rest of the package (and the
tests) can be imported without a database driver or model weights.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any, Dict, List

from .config import get_database_url


@lru_cache(maxsize=1)
def _embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-mpnet-base-v2")


@lru_cache(maxsize=1)
def _hf_api():
    from huggingface_hub import HfApi

    return HfApi()


def _connect():
    import psycopg2

    return psycopg2.connect(get_database_url(), connect_timeout=10)


def _dict_cursor(conn):
    import psycopg2.extras

    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)


# Mirrors _derive_vram(): values <= 96 are per-GPU, larger values are totals.
_TOTAL_VRAM_SQL = "(CASE WHEN gpu_memory_gb <= 96 THEN gpu_memory_gb * gpu_count ELSE gpu_memory_gb END)"

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
        query_embedding = _embedding_model().encode(norm).tolist()
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
        conn = _connect()
        cursor = _dict_cursor(conn)

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
        conn = _connect()
        cursor = _dict_cursor(conn)

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
        if min_vram_gb > 0:
            # Filter in SQL so the LIMIT below cannot cut off every match.
            query += f" AND {_TOTAL_VRAM_SQL} >= %s"
            params.append(min_vram_gb)

        if preference == "performance":
            # Largest VRAM first; ordering by price here would hide big GPUs.
            query += f" ORDER BY {_TOTAL_VRAM_SQL} DESC, price_hourly ASC LIMIT %s"
            params.append(200)
        else:
            query += " ORDER BY price_hourly ASC LIMIT %s"
            params.append(80)

        cursor.execute(query, tuple(params))
        results = cursor.fetchall()
        cursor.close()

    except Exception as e:
        return {"source": "cloud_pricing_db", "warnings": warnings, "results": [], "error": f"Database error: {str(e)}"}
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
        files = _hf_api().list_repo_files(repo_id=repo_id, repo_type="model")
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
        raw_candidates = list(_hf_api().list_models(
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
                info = _hf_api().model_info(it.id)
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


# Knowledge-base name (as used in the RetrievalPlan) -> tool function.
DEFAULT_TOOLS = {
    "cloud_pricing_db": search_cloud_infrastructure,
    "k8s_marketplace": search_packages,
    "hf_models": search_huggingface_models,
}
