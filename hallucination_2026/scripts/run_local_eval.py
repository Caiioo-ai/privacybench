"""
Local OpenAI-compatible and Ollama evaluation runner for PrivacyBench v2.1 (`hallucination_2026`).

Runs the exact same PrivacyBench v2.1 prompts, hash-verified articles, recency eligibility
rules, and result JSON schema as `run_eval.py` without modifying any hosted runner scripts
(preserving hosted `source_sha256` replay contracts byte-for-byte).

Zero API keys are read or sent: all inference calls go strictly to the local endpoint given.
Outputs are directly gradeable by `grade_eval.py --file <path>` and scoreable on the frozen
Stage-6 item set by `rollup_and_plot.py --stage6-dropped-from <rollup_summary.json>`.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from api_effort_policy import fingerprint
from llm_client import TASK_ROOT, get_article_text, is_article_eligible_for_model
from run_eval import (
    OMNISCIENCE_CLOSED_BOOK_PROMPT,
    OMNISCIENCE_GROUNDED_PROMPT,
    SUMMARY_PROMPT_V1,
    slugify_model,
)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>.*$", flags=re.DOTALL | re.IGNORECASE)
_HARMONY_FINAL_RE = re.compile(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)", flags=re.DOTALL)
_GEMMA_CHAN_BLOCK_RE = re.compile(r"<\|channel>.*?<channel\|>", flags=re.DOTALL)
_GEMMA_CHAN_OPEN_RE = re.compile(r"<\|channel>.*$", flags=re.DOTALL)


def strip_inline_think(text: str) -> Tuple[str, bool]:
    """Remove inline chain-of-thought blocks (`<think>`, Harmony `<|channel|>`, Gemma `<|channel>`)
    when a local server does not separate reasoning from content.

    If thinking was never closed (e.g. token cap exhausted mid-thought), return empty content so
    raw chain-of-thought is never graded as a final answer.
    """
    if not text:
        return "", False

    # OpenAI Harmony format (e.g. gpt-oss-20b on mlx_lm.server): extract `<|channel|>final<|message|>`.
    if "<|channel|>" in text:
        matches = _HARMONY_FINAL_RE.findall(text)
        if matches:
            return matches[-1].strip(), True
        return "", True

    # Gemma 4 control-channel format (`<|channel>thought...<channel|>`).
    if "<|channel>" in text or "<channel|>" in text:
        cleaned = _GEMMA_CHAN_BLOCK_RE.sub("", text)
        if "<|channel>" in cleaned:
            cleaned = _GEMMA_CHAN_OPEN_RE.sub("", cleaned)
        cleaned = cleaned.replace("<channel|>", "")
        return cleaned.strip(), True

    had_think = bool(re.search(r"<think>", text, flags=re.IGNORECASE))
    if not had_think:
        return text.strip(), False
    cleaned = _THINK_BLOCK_RE.sub("", text)
    if re.search(r"<think>", cleaned, flags=re.IGNORECASE):
        cleaned = _THINK_OPEN_RE.sub("", cleaned)
    return cleaned.strip(), True


def call_local_endpoint(
    *,
    endpoint: str,
    wire: str,
    model_id: str,
    served_model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int = 600,
    num_ctx: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute one local inference call with zero credentials and cost_usd=0.0."""
    base = endpoint.rstrip("/")
    t0 = time.time()

    if wire == "openai-chat":
        url = f"{base}/chat/completions"
        payload: Dict[str, Any] = {
            "model": served_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        latency = round(time.time() - t0, 3)
        if "error" in data and data["error"]:
            raise RuntimeError(f"Local endpoint error: {data['error']}")
        choices = data.get("choices") or [{}]
        msg = choices[0].get("message") or {}
        raw_content = msg.get("content") or ""
        content, had_inline_think = strip_inline_think(raw_content)
        finish_reason = choices[0].get("finish_reason") or "stop"
        usage = data.get("usage") or {}
        prompt_tok = int(usage.get("prompt_tokens") or 0)
        compl_tok = int(usage.get("completion_tokens") or 0)
        total_tok = int(usage.get("total_tokens") or (prompt_tok + compl_tok))
        details = usage.get("completion_tokens_details") or {}
        reasoning_tok = int(details.get("reasoning_tokens") or 0)
        has_separate_reasoning = bool(msg.get("reasoning") or msg.get("reasoning_content"))
        return {
            "content": content,
            "model_requested": model_id,
            "model_returned": data.get("model", served_model),
            "provider": "local:openai-chat",
            "reasoning_effort": None,
            "temperature": temperature,
            "prompt_tokens": prompt_tok,
            "completion_tokens": compl_tok,
            "reasoning_tokens": reasoning_tok,
            "total_tokens": total_tok,
            "cost_usd": 0.0,
            "latency_seconds": latency,
            "finish_reason": finish_reason,
            "reasoned": bool(has_separate_reasoning or had_inline_think or reasoning_tok > 0),
        }

    if wire == "ollama-chat":
        url = f"{base}/api/chat"
        options: Dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        if num_ctx is not None:
            options["num_ctx"] = int(num_ctx)
        payload = {
            "model": served_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": options,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        latency = round(time.time() - t0, 3)
        if "error" in data and data["error"]:
            raise RuntimeError(f"Ollama error: {data['error']}")
        msg = data.get("message") or {}
        raw_content = msg.get("content") or ""
        content, had_inline_think = strip_inline_think(raw_content)
        thinking = msg.get("thinking") or ""
        prompt_tok = int(data.get("prompt_eval_count") or 0)
        compl_tok = int(data.get("eval_count") or 0)
        done_reason = data.get("done_reason") or "stop"
        finish_reason = "length" if done_reason == "length" else "stop"
        return {
            "content": content,
            "model_requested": model_id,
            "model_returned": data.get("model", served_model),
            "provider": "local:ollama-chat",
            "reasoning_effort": None,
            "temperature": temperature,
            "prompt_tokens": prompt_tok,
            "completion_tokens": compl_tok,
            "reasoning_tokens": 0,
            "total_tokens": prompt_tok + compl_tok,
            "cost_usd": 0.0,
            "latency_seconds": latency,
            "finish_reason": finish_reason,
            "reasoned": bool(thinking or had_inline_think),
        }

    raise ValueError(f"Unsupported local wire: {wire!r}")


def _is_reusable_row(row: Dict[str, Any], task: Dict[str, Any], model_id: str, temperature: float) -> bool:
    expected = {k: v for k, v in task.items() if k not in ("prompt", "request")}
    if any(row.get(k) != v for k, v in expected.items()):
        return False
    telemetry = row.get("telemetry")
    response = row.get("response")
    if not isinstance(telemetry, dict) or not isinstance(response, str):
        return False
    if telemetry.get("model_requested") != model_id or telemetry.get("temperature") != temperature:
        return False
    return bool(
        response.strip()
        and not response.startswith("[ERROR:")
        and telemetry.get("finish_reason") != "length"
        and not telemetry.get("error")
    )


def run_local_model_evaluation(
    model_meta: Dict[str, Any],
    articles: List[Dict[str, Any]],
    items: List[Dict[str, Any]],
    results_dir: Path,
    run_date: str,
    *,
    endpoint: str,
    wire: str = "openai-chat",
    max_workers: int = 1,
    timeout: int = 600,
    num_ctx: Optional[int] = None,
    dry_run: bool = False,
) -> Path | Dict[str, Any]:
    requested_model_meta = copy.deepcopy(model_meta)
    model_meta = copy.deepcopy(model_meta)
    model_id = model_meta["model_id"]
    local_meta = model_meta.get("local") or {}
    served_model = local_meta.get("served_model") or model_id
    params = model_meta.setdefault("default_params", {})
    params["reasoning_effort"] = None
    temperature = float(params.get("temperature", 0.0))
    qa_max_tokens = int(params.get("qa_max_tokens", 16384))
    summary_max_tokens = int(params.get("summary_max_tokens", 16384))
    if not math.isfinite(temperature) or qa_max_tokens <= 0 or summary_max_tokens <= 0:
        raise ValueError("Invalid temperature/output token cap")
    params.update(temperature=temperature, qa_max_tokens=qa_max_tokens, summary_max_tokens=summary_max_tokens)
    if max_workers < 1:
        raise ValueError("workers must be positive")

    out_path = results_dir / f"{run_date}-{slugify_model(model_id)}.json"
    checkpoint_path = results_dir / f"{run_date}-{slugify_model(model_id)}.checkpoint.jsonl"
    print(
        f"\n[run_local_eval] Evaluating local model={model_id} (served_model={served_model}, wire={wire}, "
        f"effective_cutoff={model_meta['effective_cutoff']} via {model_meta['cutoff_source']})"
    )

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

    tasks: List[Dict[str, Any]] = []
    for it in eligible_items:
        aid = it["article_id"]
        sub_mode = "grounded_answerable" if it["answerable"] else "grounded_nearmiss"
        tasks.append(
            {
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
            }
        )
        if it["answerable"] and "closed_book" in it.get("modes", ["grounded_answerable", "closed_book"]):
            tasks.append(
                {
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
                }
            )

    for aid in sorted(eligible_article_ids):
        art = articles_by_id[aid]
        tasks.append(
            {
                "task_type": "summary",
                "eval_mode": "summary",
                "item_id": f"{aid}_summary",
                "article_id": aid,
                "domain": art["domain"],
                "subtopic": art["subtopic"],
                "prompt": f"{SUMMARY_PROMPT_V1}\n\n{article_texts[aid]}",
                "max_tokens": summary_max_tokens,
            }
        )

    if not tasks:
        raise ValueError(f"No eligible evaluation tasks for {model_id}")

    for task in tasks:
        task["request"] = {
            "model": served_model,
            "messages": [{"role": "user", "content": task["prompt"]}],
            "temperature": temperature,
            "max_tokens": task["max_tokens"],
            "wire": wire,
            **({"num_ctx": num_ctx} if num_ctx is not None else {}),
        }
        task["request_sha256"] = fingerprint(task["request"])
        task["prompt_sha256"] = hashlib.sha256(task["prompt"].encode("utf-8")).hexdigest()

    effort_policy = {
        "policy": {"version": "local-engine-default-v1"},
        "state": "not-applicable-local-default",
        "selected_effort": None,
        "local": {
            "engine": local_meta.get("engine", wire),
            "served_model": served_model,
            "quantization": local_meta.get("quantization", "unknown"),
            "wire": wire,
            "num_ctx": num_ctx,
        },
    }
    effort_policy["fingerprint"] = fingerprint(effort_policy)

    source_sha256 = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in ("run_local_eval.py", "run_eval.py", "llm_client.py")
    }
    contract = {
        "version": "local-exact-request-v1",
        "condition": f"local-{wire}-engine-default-v1",
        "requested_model_meta": requested_model_meta,
        "effective_model_meta": model_meta,
        "effort_policy": effort_policy,
        "source_sha256": source_sha256,
        "corpus_sha256": {"articles": fingerprint(articles), "items": fingerprint(items)},
        "tasks": [{k: v for k, v in t.items() if k not in ("prompt", "request")} for t in tasks],
    }
    contract["fingerprint"] = fingerprint(contract)

    expected_tasks = {f"{t['eval_mode']}::{t['item_id']}": t for t in tasks}
    existing_cache: Dict[str, Dict[str, Any]] = {}

    if out_path.exists():
        old_data = json.loads(out_path.read_text(encoding="utf-8"))
        if not isinstance(old_data, dict) or old_data.get("inference_contract") != contract:
            raise ValueError("Cache refused: legacy or incompatible local inference contract; use a new output path")
        for group in ("qa_responses", "summary_responses"):
            for row in old_data.get(group, []):
                key = f"{row.get('eval_mode')}::{row.get('item_id')}"
                task = expected_tasks.get(key)
                if task and _is_reusable_row(row, task, model_id, temperature):
                    existing_cache[key] = row

    if checkpoint_path.exists():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("contract_fingerprint") != contract["fingerprint"]:
                continue
            row = rec.get("row") or {}
            key = f"{row.get('eval_mode')}::{row.get('item_id')}"
            task = expected_tasks.get(key)
            if task and _is_reusable_row(row, task, model_id, temperature):
                existing_cache[key] = row

    plan = {
        "model_id": model_id,
        "served_model": served_model,
        "condition": contract["condition"],
        "resolved_params": params,
        "inference_fingerprint": contract["fingerprint"],
        "planned_calls": len(tasks),
        "reusable_calls": len(existing_cache),
        "output": str(out_path),
    }
    if dry_run:
        return plan
    if len(existing_cache) == len(tasks) and out_path.exists():
        return out_path

    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_lock = threading.Lock()

    def _exec_one(t: Dict[str, Any]) -> Dict[str, Any]:
        key = f"{t['eval_mode']}::{t['item_id']}"
        if key in existing_cache:
            return existing_cache[key]
        try:
            resp = call_local_endpoint(
                endpoint=endpoint,
                wire=wire,
                model_id=model_id,
                served_model=served_model,
                prompt=t["prompt"],
                temperature=temperature,
                max_tokens=t["max_tokens"],
                timeout=timeout,
                num_ctx=num_ctx,
            )
            row = {
                **{k: v for k, v in t.items() if k not in ("prompt", "request")},
                "response": resp["content"],
                "telemetry": {k: v for k, v in resp.items() if k != "content"},
            }
            if _is_reusable_row(row, t, model_id, temperature):
                with ckpt_lock:
                    with checkpoint_path.open("a", encoding="utf-8") as fh:
                        fh.write(
                            json.dumps(
                                {"contract_fingerprint": contract["fingerprint"], "row": row},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
            return row
        except Exception as e:
            return {
                **{k: v for k, v in t.items() if k not in ("prompt", "request")},
                "response": f"[ERROR: {type(e).__name__}: {str(e)[:160]}]",
                "telemetry": {
                    "model_requested": model_id,
                    "model_returned": served_model,
                    "provider": f"local:{wire}:error",
                    "reasoning_effort": None,
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

    results_list: List[Dict[str, Any]] = [None] * len(tasks)  # type: ignore[list-item]
    t_start = time.time()
    done_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_to_idx = {ex.submit(_exec_one, tasks[i]): i for i in range(len(tasks))}
        for fut in concurrent.futures.as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            results_list[idx] = fut.result()
            done_count += 1
            if done_count % 15 == 0 or done_count == len(tasks):
                print(f"  [{model_id}] completed {done_count}/{len(tasks)} calls ({time.time() - t_start:.1f}s)")

    results_list = [{k: v for k, v in r.items() if k != "grading"} for r in results_list]
    qa_responses = [r for r in results_list if r["task_type"] == "qa"]
    summary_responses = [r for r in results_list if r["task_type"] == "summary"]
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
            "total_cost_usd": 0.0,
            "wall_time_seconds": round(time.time() - t_start, 2),
        },
        "qa_responses": qa_responses,
        "summary_responses": summary_responses,
    }
    temporary = out_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(out_path)
    if checkpoint_path.exists():
        errors = [
            r for r in results_list if r["response"].startswith("[ERROR:") or r["telemetry"].get("error")
        ]
        if not errors:
            checkpoint_path.unlink()
    print(f"  -> Saved {out_path} (tokens={total_tokens}, cost=$0.0000)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run PrivacyBench v2.1 against a local model endpoint.")
    ap.add_argument("--model", required=True, help="model_id from local_models.json")
    ap.add_argument("--endpoint", required=True, help="Base URL, e.g. http://127.0.0.1:18787/v1 or http://127.0.0.1:11434")
    ap.add_argument("--wire", choices=("openai-chat", "ollama-chat"), default="openai-chat")
    ap.add_argument("--models-file", default=str(TASK_ROOT / "evaluation_questions" / "local_models.json"))
    ap.add_argument("--articles-file", default=str(TASK_ROOT / "evaluation_questions" / "articles.json"))
    ap.add_argument("--items-file", default=str(TASK_ROOT / "evaluation_questions" / "items.json"))
    ap.add_argument("--results-dir", default=str(TASK_ROOT / "results" / "local"))
    ap.add_argument("--run-date", default="2026-11-21")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--num-ctx", type=int, default=None, help="Explicit Ollama num_ctx (e.g. 32768)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    models_data = json.loads(Path(args.models_file).read_text(encoding="utf-8"))["models"]
    articles = json.loads(Path(args.articles_file).read_text(encoding="utf-8"))["articles"]
    items = json.loads(Path(args.items_file).read_text(encoding="utf-8"))["items"]
    matches = [m for m in models_data if m["model_id"] == args.model]
    if not matches:
        raise ValueError(f"Model {args.model!r} not found in {args.models_file}")

    res = run_local_model_evaluation(
        matches[0],
        articles,
        items,
        Path(args.results_dir),
        args.run_date,
        endpoint=args.endpoint,
        wire=args.wire,
        max_workers=args.workers,
        timeout=args.timeout,
        num_ctx=args.num_ctx,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
