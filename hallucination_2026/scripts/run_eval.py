"""
Benchmark evaluation runner for PrivacyBench v2 (`hallucination_2026`).

Evaluates target models across three modes per eligible article:
- Mode A (`grounded_answerable` and `grounded_nearmiss`): Article in context
- Mode B (`closed_book`): Standalone answerable questions without article or tools
- Mode C (`summary`): 5-sentence article summarization ("Please summarize the following news article in 5 sentences.")

Enforces model-relative recency gating:
An `(article, model)` pair is eligible iff `article.published_at > model.effective_cutoff`.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import math
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from api_effort_policy import fetch_capability_catalog, resolve_effort, resolve_level_effort, fingerprint
import cometapi_client

from llm_client import (
    TASK_ROOT,
    call_llm,
    cometapi_routes,
    gateway_name,
    loaded_route,
    use_cometapi,
    get_article_text,
    is_article_eligible_for_model,
)

# Adapted from AA-Omniscience (arXiv:2511.13029 / ArtificialAnalysis/AA-Omniscience-Public, Apache-2.0)
OMNISCIENCE_CLOSED_BOOK_PROMPT = """You are answering questions about {domain}, and in particular {subtopic}.
You will be given a question, answer with JUST the answer (no explanation).
If you do not know the answer, or you need more context or tools to answer the question,
be clear about this - it is better that you say this than get the wrong answer.

Question: {question}"""

OMNISCIENCE_GROUNDED_PROMPT = """You are answering questions about {domain}, and in particular {subtopic}, using ONLY the source article provided below.
You will be given a question, answer with JUST the answer (no explanation).
Use only facts explicitly stated in the article. Do not extrapolate or guess from adjacent figures.
If the article does not state the answer to the question, or you need more context to answer accurately,
be clear about this (for example, reply "Not stated in the article") - it is better that you say this than get the wrong answer.

Source Article:
{article_text}

Question: {question}"""

# Preserved verbatim from bias_study (v1) for continuity
SUMMARY_PROMPT_V1 = "Please summarize the following news article in 5 sentences."


def slugify_model(model_id: str) -> str:
    return model_id.replace("/", "-").replace(":", "-")


def run_model_evaluation(
    model_meta: Dict[str, Any],
    articles: List[Dict[str, Any]],
    items: List[Dict[str, Any]],
    results_dir: Path,
    run_date: str,
    max_workers: int = 8,
    *,
    capability_catalog: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
    effort_level: Optional[str] = None,
    max_model_cost_usd: Optional[float] = None,
) -> Path | Dict[str, Any]:
    requested_model_meta = copy.deepcopy(model_meta)
    model_meta = copy.deepcopy(model_meta)
    model_id = model_meta["model_id"]
    catalog = capability_catalog if capability_catalog is not None else fetch_capability_catalog()
    # One level for every model (the router's snap), or the legacy highest declared native effort.
    effort_policy = (resolve_level_effort(model_id, catalog, effort_level) if effort_level
                     else resolve_effort(model_id, catalog))
    gateway = gateway_name()
    route = loaded_route(model_id)
    explicit_cache = route is not None
    params = model_meta.setdefault("default_params", {})
    params["reasoning_effort"] = effort_policy["selected_effort"]
    temperature = float(params.get("temperature", 0.0))
    qa_max_tokens = int(params.get("qa_max_tokens", 256))
    summary_max_tokens = int(params.get("summary_max_tokens", 600))
    reasoning_effort = params["reasoning_effort"]
    if not math.isfinite(temperature) or qa_max_tokens <= 0 or summary_max_tokens <= 0:
        raise ValueError("Invalid temperature/output token cap")
    params.update(temperature=temperature, qa_max_tokens=qa_max_tokens, summary_max_tokens=summary_max_tokens)
    if max_workers < 1:
        raise ValueError("workers must be positive")

    out_path = results_dir / f"{run_date}-{slugify_model(model_id)}.json"
    print(f"\n[run_eval] Evaluating model={model_id} (effective_cutoff={model_meta['effective_cutoff']} via {model_meta['cutoff_source']})")

    articles_by_id = {a["id"]: a for a in articles}
    article_texts: Dict[str, str] = {}
    eligibility_map: Dict[str, Dict[str, Any]] = {}

    for art in articles:
        aid = art["id"]
        eligible, reason = is_article_eligible_for_model(art, model_meta)
        eligibility_map[aid] = {
            "article_id": aid,
            "published_at": art["published_at"],
            "eligible": eligible,
            "reason": reason,
        }
        if eligible:
            article_texts[aid] = get_article_text(art, verify_hash=True)

    eligible_article_ids = {aid for aid, info in eligibility_map.items() if info["eligible"]}
    eligible_items = [it for it in items if it["article_id"] in eligible_article_ids]
    print(
        f"  Eligible articles: {len(eligible_article_ids)}/{len(articles)} | "
        f"Eligible QA items: {len(eligible_items)}/{len(items)}"
    )

    # Build tasks:
    # 1. Grounded QA (all eligible items: both answerable and near-miss)
    # 2. Closed-book QA (all eligible answerable items)
    # 3. 5-sentence Summary (all eligible articles)
    tasks: List[Dict[str, Any]] = []
    for it in eligible_items:
        aid = it["article_id"]
        sub_mode = "grounded_answerable" if it["answerable"] else "grounded_nearmiss"
        tasks.append({
            "task_type": "qa",
            "eval_mode": sub_mode,
            "item_id": it["id"],
            "article_id": aid,
            "domain": it["domain"],
            "subtopic": it["subtopic"],
            "question": it["question"],
            "gold_answer": it["gold_answer"],
            "answer_type": it["answer_type"],
            "numeric_precision_rule": it["numeric_precision_rule"],
            "answerable": it["answerable"],
            "generator_model": it["generator_model"],
            "prompt": OMNISCIENCE_GROUNDED_PROMPT.format(
                domain=it["domain"],
                subtopic=it["subtopic"],
                article_text=article_texts[aid],
                question=it["question"],
            ),
            "max_tokens": qa_max_tokens,
        })
        if it["answerable"] and "closed_book" in it.get("modes", ["grounded_answerable", "closed_book"]):
            tasks.append({
                "task_type": "qa",
                "eval_mode": "closed_book",
                "item_id": it["id"],
                "article_id": aid,
                "domain": it["domain"],
                "subtopic": it["subtopic"],
                "question": it["question"],
                "gold_answer": it["gold_answer"],
                "answer_type": it["answer_type"],
                "numeric_precision_rule": it["numeric_precision_rule"],
                "answerable": True,
                "generator_model": it["generator_model"],
                "prompt": OMNISCIENCE_CLOSED_BOOK_PROMPT.format(
                    domain=it["domain"],
                    subtopic=it["subtopic"],
                    question=it["question"],
                ),
                "max_tokens": qa_max_tokens,
            })

    for aid in sorted(eligible_article_ids):
        art = articles_by_id[aid]
        tasks.append({
            "task_type": "summary",
            "eval_mode": "summary",
            "item_id": f"{aid}_summary",
            "article_id": aid,
            "domain": art["domain"],
            "subtopic": art["subtopic"],
            "prompt": f"{SUMMARY_PROMPT_V1}\n\n{article_texts[aid]}",
            "max_tokens": summary_max_tokens,
        })

    if not tasks:
        raise ValueError(f"No eligible evaluation tasks for {model_id}")
    for task in tasks:
        if explicit_cache and task["eval_mode"].startswith("grounded"):
            # Everything before the question is one article's shared text: the cacheable prefix, unchanged prompt.
            task["cache_prefix_chars"] = task["prompt"].rindex("Question: ")
            task["cache_key"] = f"privacybench-{task['article_id']}"
        task["request"] = {"model": model_id, "messages": [{"role": "user", "content": task["prompt"]}],
                           "reasoning_effort": reasoning_effort, "temperature": temperature,
                           "max_tokens": task["max_tokens"]}
        if explicit_cache:
            task["request"].update(gateway=gateway, cache_prefix_chars=task.get("cache_prefix_chars"),
                                   cache_key=task.get("cache_key"))
        task["request_sha256"] = fingerprint(task["request"])
        task["prompt_sha256"] = hashlib.sha256(task["prompt"].encode()).hexdigest()
    source_sha256 = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                     for name in ("run_eval.py", "api_effort_policy.py", "llm_client.py", "cometapi_client.py")}
    # Retrieval time / whole-catalogue churn are receipts, not native capability changes.
    policy_identity = {k: v for k, v in effort_policy.items() if k != "catalog_retrieval"}
    condition = f"{gateway}-api-" + (f"level-{effort_level}-v1" if effort_level else "highest-native-v1")
    contract = {"version": "api-exact-request-v1", "condition": condition,
                **({"gateway_route": route} if route is not None else {}),
                "requested_model_meta": requested_model_meta, "effective_model_meta": model_meta,
                "effort_policy": policy_identity, "source_sha256": source_sha256,
                "corpus_sha256": {"articles": fingerprint(articles), "items": fingerprint(items)},
                "tasks": [{k: v for k, v in t.items() if k not in ("prompt", "request")}
                          for t in tasks]}
    contract["fingerprint"] = fingerprint(contract)
    existing_cache: Dict[str, Dict[str, Any]] = {}
    old_data = None
    expected_tasks = {f"{t['eval_mode']}::{t['item_id']}": t for t in tasks}
    if len(expected_tasks) != len(tasks):
        raise ValueError("Duplicate task identity")
    if out_path.exists():
        try:
            old_data = json.loads(out_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise ValueError("Cache refused: unreadable existing output; preserve it and use a new path") from exc
        if not isinstance(old_data, dict) or old_data.get("inference_contract") != contract:
            raise ValueError("Cache refused: legacy or incompatible inference contract; use a new output path")
        if old_data.get("model_meta") != model_meta:
            raise ValueError("Cache refused: effective model metadata differs")
        seen = set()
        for group, kind in (("qa_responses", "qa"), ("summary_responses", "summary")):
            rows = old_data.get(group)
            if not isinstance(rows, list):
                raise ValueError("Cache refused: malformed response collection")
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("Cache refused: malformed response")
                key = f"{row.get('eval_mode')}::{row.get('item_id')}"
                task = expected_tasks.get(key)
                if task is None or key in seen or task["task_type"] != kind:
                    raise ValueError("Cache refused: unknown/duplicate/misplaced response")
                seen.add(key)
                expected = {k: v for k, v in task.items() if k not in ("prompt", "request")}
                if any(row.get(k) != v for k, v in expected.items()):
                    raise ValueError("Cache refused: cached request/item differs")
                telemetry = row.get("telemetry")
                response = row.get("response")
                if not isinstance(telemetry, dict) or not isinstance(response, str):
                    raise ValueError("Cache refused: malformed response/telemetry")
                if (telemetry.get("model_requested") != model_id or
                        telemetry.get("reasoning_effort") != reasoning_effort or
                        telemetry.get("temperature") != temperature):
                    raise ValueError("Cache refused: cached delivery settings differ")
                if (response.strip() and not response.startswith("[ERROR:") and
                        telemetry.get("finish_reason") != "length" and not telemetry.get("error")):
                    existing_cache[key] = row
    plan = {"model_id": model_id, "condition": condition, "gateway_route": route,
            "requested_params": requested_model_meta.get("default_params", {}),
            "resolved_params": params, "effort_policy": effort_policy,
            "inference_fingerprint": contract["fingerprint"], "planned_calls": len(tasks),
            "reusable_calls": len(existing_cache), "provider_calls": 0,
            "output": str(out_path), "max_attempts_per_item": 1}
    if dry_run:
        return plan
    # A fully unchanged replay preserves the entire graded result byte for byte.
    if len(existing_cache) == len(tasks):
        return out_path

    spent = {"usd": 0.0}
    spend_lock = threading.Lock()

    def _exec_one(t: Dict[str, Any]) -> Dict[str, Any]:
        cache_key = f"{t['eval_mode']}::{t['item_id']}"
        if cache_key in existing_cache:
            return existing_cache[cache_key]
        try:
            with spend_lock:
                if max_model_cost_usd is not None and spent["usd"] >= max_model_cost_usd:
                    raise RuntimeError(f"SpendCap: ${spent['usd']:.4f} of the ${max_model_cost_usd:.2f} cap spent")
            prefix_chars = t.get("cache_prefix_chars")
            resp = call_llm(
                model_id=model_id,
                messages=[{"role": "user", "content": t["prompt"]}],
                temperature=temperature,
                max_tokens=t["max_tokens"],
                reasoning_effort=reasoning_effort,
                # High-effort calls through a gateway queue far past the OpenRouter-era 90 s.
                timeout=300 if explicit_cache else 90,
                # The shared client otherwise silently doubles the output cap on retry.
                max_retries=1,
                cache_prefix=t["prompt"][:prefix_chars] if prefix_chars else None,
                cache_key=t.get("cache_key"),
            )
            with spend_lock:
                spent["usd"] += float(resp.get("cost_usd") or 0.0)
            if (resp.get("model_requested") != model_id or resp.get("reasoning_effort") != reasoning_effort
                    or resp.get("temperature") != temperature):
                raise ValueError("Provider request telemetry differs from resolved settings")
            return {**{k: v for k, v in t.items() if k not in ("prompt", "request")}, "response": resp["content"], "telemetry": {k: v for k, v in resp.items() if k != "content"}}
        except Exception as e:
            return {
                **{k: v for k, v in t.items() if k not in ("prompt", "request")},
                "response": f"[ERROR: {type(e).__name__}: {str(e)[:160]}]",
                "telemetry": {
                    "model_requested": model_id,
                    "model_returned": model_id,
                    "provider": "error",
                    "reasoning_effort": reasoning_effort,
                    "temperature": temperature,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "reasoning_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "latency_seconds": 0.0,
                    "error": str(e),
                },
            }

    results_list: List[Dict[str, Any]] = [None] * len(tasks)  # type: ignore
    t_start = time.time()
    # Warm, then fan out: each article's first cached call finishes before its siblings are sent, so they can read
    # the prefix it wrote instead of all writing it at once. Ordering only; every request is unchanged.
    warm = sorted({t["cache_key"]: i for i, t in reversed(list(enumerate(tasks))) if t.get("cache_key")}.values())
    phases = [warm, [i for i in range(len(tasks)) if i not in set(warm)]]
    done_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for phase in phases:
            fut_to_idx = {ex.submit(_exec_one, tasks[i]): i for i in phase}
            for fut in concurrent.futures.as_completed(fut_to_idx):
                idx = fut_to_idx[fut]
                results_list[idx] = fut.result()
                done_count += 1
                if done_count % 25 == 0 or done_count == len(tasks):
                    print(f"  [{model_id}] completed {done_count}/{len(tasks)} calls ({time.time() - t_start:.1f}s)")

    # Any new inference invalidates prior panel outcomes, including cached row grades.
    results_list = [{k: v for k, v in r.items() if k != "grading"} for r in results_list]
    qa_responses = [r for r in results_list if r["task_type"] == "qa"]
    summary_responses = [r for r in results_list if r["task_type"] == "summary"]

    total_cost = round(sum(r["telemetry"].get("cost_usd", 0.0) for r in results_list), 6)
    total_tokens = sum(r["telemetry"].get("total_tokens", 0) for r in results_list)

    payload = {
        "run_date": run_date,
        "model_meta": model_meta,
        "effort_policy": effort_policy,
        "inference_contract": contract,
        "eligibility": {
            "total_articles": len(articles),
            "eligible_articles": len(eligible_article_ids),
            "ineligible_articles": len(articles) - len(eligible_article_ids),
            "total_items_in_corpus": len(items),
            "eligible_qa_items": len(eligible_items),
            "per_article": eligibility_map,
        },
        "inference_totals": {
            "total_calls": len(results_list),
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "wall_time_seconds": round(time.time() - t_start, 2),
        },
        "qa_responses": qa_responses,
        "summary_responses": summary_responses,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(out_path)
    print(f"  -> Saved {out_path} (tokens={total_tokens}, cost=${total_cost:.4f})")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run PrivacyBench v2 evaluation for one or more models.")
    ap.add_argument("--model", default=None, help="Single model_id from models.json to evaluate")
    ap.add_argument("--all-pilot-models", action="store_true", help="Evaluate all models in models.json")
    ap.add_argument("--models-file", default=str(TASK_ROOT / "evaluation_questions" / "models.json"))
    ap.add_argument("--articles-file", default=str(TASK_ROOT / "evaluation_questions" / "articles.json"))
    ap.add_argument("--items-file", default=str(TASK_ROOT / "evaluation_questions" / "items.json"))
    ap.add_argument("--results-dir", default=str(TASK_ROOT / "results"))
    ap.add_argument("--run-date", default="2026-10-15")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", "--preflight", action="store_true",
                    help="Fetch current public capabilities and validate exact cache/settings without inference")
    ap.add_argument("--models", default=None, help="Comma-separated model_ids from models.json to evaluate")
    ap.add_argument("--gateway", choices=("openrouter", cometapi_client.GATEWAY), default="openrouter",
                    help="cometapi: each model on its registry route, with prompt caching")
    ap.add_argument("--effort-level", default=None,
                    help="Run every model at this level, snapped to its declared levels as the Caiioo router snaps it "
                         "(default: the highest declared native effort)")
    ap.add_argument("--max-model-cost-usd", type=float, default=None,
                    help="Stop sending a model's calls once its run has cost this much (later calls record an error)")
    args = ap.parse_args()

    models_data = json.loads(Path(args.models_file).read_text(encoding="utf-8"))["models"]
    articles = json.loads(Path(args.articles_file).read_text(encoding="utf-8"))["articles"]
    items = json.loads(Path(args.items_file).read_text(encoding="utf-8"))["items"]
    results_dir = Path(args.results_dir)
    if args.all_pilot_models:
        target_models = models_data
    elif args.models:
        wanted = [m.strip() for m in args.models.split(",") if m.strip()]
        by_id = {m["model_id"]: m for m in models_data}
        missing = [m for m in wanted if m not in by_id]
        if missing:
            raise ValueError(f"Models {missing} not found in {args.models_file}")
        target_models = [by_id[m] for m in wanted]
    elif args.model:
        target_models = [m for m in models_data if m["model_id"] == args.model]
        if not target_models:
            raise ValueError(f"Model {args.model!r} not found in {args.models_file}")
    else:
        raise ValueError("Specify --model <id> or --all-pilot-models")

    catalog = fetch_capability_catalog()
    if args.gateway == cometapi_client.GATEWAY:
        use_cometapi(cometapi_routes(Path(args.models_file), [m["model_id"] for m in target_models],
                                     cometapi_client.fetch_catalog()))
    run_options = dict(max_workers=args.workers, capability_catalog=catalog, effort_level=args.effort_level,
                       max_model_cost_usd=args.max_model_cost_usd)
    # Validate ALL targets/caches before the first potentially paid call.
    plans = [run_model_evaluation(m, articles, items, results_dir, args.run_date, dry_run=True, **run_options)
             for m in target_models]
    print(json.dumps({"dry_run": args.dry_run, "plans": plans}, indent=2, ensure_ascii=False))
    if args.dry_run:
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, len(target_models))) as executor:
        futures = [executor.submit(run_model_evaluation, m, articles, items, results_dir, args.run_date, **run_options)
                   for m in target_models]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
