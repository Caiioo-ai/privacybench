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

# Caiioo production loop detector patterns (ported from src/shared/useTerminalRuntime.ts + src/shared/utils/output-safety.ts)
_IDENTICAL_CHAR_RUN_RE = re.compile(r"(.)\1{127,}", flags=re.DOTALL)
_TABLE_SEP_LINE_RE = re.compile(r"^[\s|:-]+$")
_TABLE_CELL_SEP_RE = re.compile(r"^:?-{3,}:?$")
_TABLE_CELL_STREAM_SEP_RE = re.compile(r"^:?-*:?$")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]+\)")
_BARE_URL_RE = re.compile(r"https?://[^\s)>\]]+")
_FENCED_CODE_RE = re.compile(r"```[\s\S]*?```")
_ALPHA_10_RE = re.compile(r"[a-zA-Z]{10,}")
_INTRA_BLOCK_SIZES = (200, 180, 160, 140, 120, 100, 80, 60, 50, 40, 30, 25, 20, 15, 12, 10, 8)


def _mask_table_delimiters(text: str) -> str:
    """Keep markdown alignment syntax out of prose-loop detection (mirrors Caiioo output-safety.ts)."""
    lines = text.split("\n")

    def _cells_of(line: str) -> List[str]:
        s = line.strip()
        if s.startswith("|"):
            s = s[1:]
        if s.endswith("|") and not s.endswith("\\|"):
            s = s[:-1]
        return [c.strip() for c in re.split(r"(?<!\\)\|", s)]

    out: List[str] = []
    for idx, line in enumerate(lines):
        if idx == 0 or not _TABLE_SEP_LINE_RE.match(line):
            out.append(line)
            continue
        header = _cells_of(lines[idx - 1])
        if len(header) < 2 or not any(re.search(r"[^\s|:-]", c) for c in header):
            out.append(line)
            continue
        cells = _cells_of(line)
        streaming_tail = idx == len(lines) - 1
        if len(cells) > len(header) or (not streaming_tail and len(cells) != len(header)):
            out.append(line)
            continue
        valid = True
        for ci, cell in enumerate(cells):
            pat = _TABLE_CELL_STREAM_SEP_RE if (streaming_tail and ci == len(cells) - 1) else _TABLE_CELL_SEP_RE
            if not pat.match(cell):
                valid = False
                break
        out.append(" " * len(line) if valid else line)
    return "\n".join(out)


def detect_generation_loop(content: str) -> Optional[str]:
    """Detect degenerate repetition loops during streaming generation (Caiioo parity).

    Returns the matched loop pattern name (`identical_char_run`, `intra_message_repetition`,
    or `repeated_phrase`), or None when output is progressing normally.
    """
    if not content or len(content) < 40:
        return None

    # Pattern 1: 128+ consecutive identical characters (useTerminalRuntime.ts)
    tail_800 = content[-800:]
    if _IDENTICAL_CHAR_RUN_RE.search(tail_800):
        return "identical_char_run"

    # Pattern 2: Tight intra-message verbatim block repetition (output-safety.ts)
    raw_tail = content[-1500:]
    prose_tail = raw_tail
    if any("|" in ln and "-" in ln and _TABLE_SEP_LINE_RE.match(ln) for ln in raw_tail.split("\n")):
        tail_start = len(content) - len(raw_tail)
        first_ln = content.rfind("\n", 0, tail_start) + 1 if tail_start > 0 else 0
        ctx_start = content.rfind("\n", 0, max(0, first_ln - 1)) + 1 if first_ln > 1 else 0
        prose_tail = _mask_table_delimiters(content[ctx_start:])[-len(raw_tail):]

    tail_clean = _BARE_URL_RE.sub("", _MD_LINK_RE.sub(r"\1", prose_tail))
    for block_size in _INTRA_BLOCK_SIZES:
        req_count = 3 if block_size >= 20 else (4 if block_size >= 12 else 5)
        if len(tail_clean) < block_size * req_count:
            continue
        block = tail_clean[-block_size:]
        stripped = block.strip()
        if not stripped or len(set(stripped)) == 1:
            continue
        indices: List[int] = []
        search_from = 0
        limit = len(tail_clean) - block_size
        while search_from <= limit:
            pos = tail_clean.find(block, search_from)
            if pos == -1:
                break
            indices.append(pos)
            search_from = pos + 1
        if len(indices) >= req_count:
            span = (indices[-1] + block_size) - indices[0]
            density = (len(indices) * block_size) / max(span, 1)
            min_density = 0.45 if block_size >= 50 else 0.65
            if density >= min_density:
                return "intra_message_repetition"

    # Pattern 3: Non-overlapping 50+ char phrase repeated 5+ times (useTerminalRuntime.ts)
    if len(tail_800) >= 250:
        cleaned = _FENCED_CODE_RE.sub(" ", tail_800)
        cleaned = re.sub(r"\|[-:\s]+\|", " ", cleaned)
        phrase_len = 50
        if len(cleaned) >= phrase_len * 5:
            for i in range(0, len(cleaned) - phrase_len * 5 + 1, 5):
                phrase = cleaned[i : i + phrase_len]
                if not _ALPHA_10_RE.search(phrase) or len(phrase.strip()) < phrase_len // 2:
                    continue
                count = 0
                pos = 0
                while True:
                    hit = cleaned.find(phrase, pos)
                    if hit == -1:
                        break
                    count += 1
                    pos = hit + phrase_len
                    if count >= 5:
                        return "repeated_phrase"
    return None


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


def _tps(tokens: int, seconds: float) -> Optional[float]:
    if tokens <= 0 or seconds <= 0:
        return None
    return round(tokens / seconds, 2)


def _check_loop(thinking: str, raw_content: str) -> Tuple[Optional[str], Optional[str]]:
    """Check both reasoning and visible content channels for Caiioo repetition loops."""
    pat = detect_generation_loop(thinking)
    if pat:
        return pat, "thinking"
    pat = detect_generation_loop(raw_content)
    if pat:
        return pat, "content"
    return None, None


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
    """Execute one local inference call with zero credentials, streaming loop detection, and cost_usd=0.0."""
    base = endpoint.rstrip("/")
    t0 = time.time()

    if wire == "openai-chat":
        url = f"{base}/chat/completions"
        payload: Dict[str, Any] = {
            "model": served_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        raw_content_parts: List[str] = []
        thinking_parts: List[str] = []
        model_returned = served_model
        finish_reason = "stop"
        prompt_tok = 0
        compl_tok = 0
        total_tok = 0
        reasoning_tok = 0
        loop_pattern: Optional[str] = None
        loop_channel: Optional[str] = None
        chars_since_check = 0
        streamed_chunks = 0

        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if hasattr(resp, "__iter__"):
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip() if isinstance(raw_line, (bytes, bytearray)) else str(raw_line).strip()
                    if not line:
                        continue
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        chunk = json.loads(data_str)
                    elif line.startswith("{"):
                        # Fallback: server/test returned a single non-streaming JSON response
                        chunk = json.loads(line)
                    else:
                        continue
                    if "error" in chunk and chunk["error"]:
                        raise RuntimeError(f"Local endpoint error: {chunk['error']}")
                    if chunk.get("model"):
                        model_returned = chunk["model"]
                    usage = chunk.get("usage")
                    if isinstance(usage, dict) and usage:
                        prompt_tok = int(usage.get("prompt_tokens") or prompt_tok)
                        compl_tok = int(usage.get("completion_tokens") or compl_tok)
                        total_tok = int(usage.get("total_tokens") or (prompt_tok + compl_tok))
                        details = usage.get("completion_tokens_details") or {}
                        reasoning_tok = int(details.get("reasoning_tokens") or reasoning_tok)
                    choices = chunk.get("choices") or []
                    if choices:
                        c0 = choices[0]
                        if c0.get("finish_reason"):
                            finish_reason = c0["finish_reason"]
                        delta = c0.get("delta") or c0.get("message") or {}
                        dc = delta.get("content") or ""
                        dr = delta.get("reasoning") or delta.get("reasoning_content") or ""
                        if dc:
                            raw_content_parts.append(dc)
                            chars_since_check += len(dc)
                            streamed_chunks += 1
                        if dr:
                            thinking_parts.append(dr)
                            chars_since_check += len(dr)
                            streamed_chunks += 1
                        if chars_since_check >= 64:
                            chars_since_check = 0
                            lp, lc = _check_loop("".join(thinking_parts), "".join(raw_content_parts))
                            if lp:
                                loop_pattern, loop_channel = lp, lc
                                finish_reason = "loop"
                                if hasattr(resp, "close"):
                                    resp.close()
                                break
            else:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
                if "error" in data and data["error"]:
                    raise RuntimeError(f"Local endpoint error: {data['error']}")
                model_returned = data.get("model", served_model)
                choices = data.get("choices") or [{}]
                msg = choices[0].get("message") or {}
                raw_content_parts.append(msg.get("content") or "")
                thinking_parts.append(msg.get("reasoning") or msg.get("reasoning_content") or "")
                finish_reason = choices[0].get("finish_reason") or "stop"
                usage = data.get("usage") or {}
                prompt_tok = int(usage.get("prompt_tokens") or 0)
                compl_tok = int(usage.get("completion_tokens") or 0)
                total_tok = int(usage.get("total_tokens") or (prompt_tok + compl_tok))
                details = usage.get("completion_tokens_details") or {}
                reasoning_tok = int(details.get("reasoning_tokens") or 0)

        raw_elapsed = time.time() - t0
        latency = round(raw_elapsed, 3)
        raw_content = "".join(raw_content_parts)
        thinking = "".join(thinking_parts)
        if loop_pattern is None:
            lp, lc = _check_loop(thinking, raw_content)
            if lp:
                loop_pattern, loop_channel = lp, lc
                finish_reason = "loop"
        if loop_pattern is not None:
            content = ""
            had_inline_think = bool(re.search(r"<think>|<\|channel", raw_content, flags=re.IGNORECASE))
            if compl_tok == 0:
                compl_tok = max(streamed_chunks, (len(raw_content) + len(thinking)) // 4)
            if total_tok == 0:
                total_tok = prompt_tok + compl_tok
        else:
            content, had_inline_think = strip_inline_think(raw_content)
            if total_tok == 0:
                total_tok = prompt_tok + compl_tok
        return {
            "content": content,
            "model_requested": model_id,
            "model_returned": model_returned,
            "provider": "local:openai-chat",
            "reasoning_effort": None,
            "temperature": temperature,
            "prompt_tokens": prompt_tok,
            "completion_tokens": compl_tok,
            "reasoning_tokens": reasoning_tok,
            "total_tokens": total_tok,
            "cost_usd": 0.0,
            "latency_seconds": latency,
            "tokens_per_second": _tps(compl_tok, max(raw_elapsed, 1e-6)),
            "finish_reason": finish_reason,
            "reasoned": bool(thinking or had_inline_think or reasoning_tok > 0),
            **({"loop_detected": True, "loop_pattern": loop_pattern, "loop_channel": loop_channel} if loop_pattern else {}),
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
            "stream": True,
            "options": options,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        raw_content_parts = []
        thinking_parts = []
        model_returned = served_model
        done_reason = "stop"
        prompt_tok = 0
        compl_tok = 0
        prompt_eval_ns = 0
        eval_ns = 0
        loop_pattern = None
        loop_channel = None
        chars_since_check = 0
        streamed_chunks = 0

        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if hasattr(resp, "__iter__"):
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip() if isinstance(raw_line, (bytes, bytearray)) else str(raw_line).strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if "error" in data and data["error"]:
                        raise RuntimeError(f"Ollama error: {data['error']}")
                    if data.get("model"):
                        model_returned = data["model"]
                    msg = data.get("message") or {}
                    dc = msg.get("content") or ""
                    dt = msg.get("thinking") or ""
                    if dc:
                        raw_content_parts.append(dc)
                        chars_since_check += len(dc)
                        streamed_chunks += 1
                    if dt:
                        thinking_parts.append(dt)
                        chars_since_check += len(dt)
                        streamed_chunks += 1
                    if data.get("done"):
                        prompt_tok = int(data.get("prompt_eval_count") or prompt_tok)
                        compl_tok = int(data.get("eval_count") or compl_tok)
                        prompt_eval_ns = int(data.get("prompt_eval_duration") or 0)
                        eval_ns = int(data.get("eval_duration") or 0)
                        done_reason = data.get("done_reason") or "stop"
                        break
                    if chars_since_check >= 64:
                        chars_since_check = 0
                        lp, lc = _check_loop("".join(thinking_parts), "".join(raw_content_parts))
                        if lp:
                            loop_pattern, loop_channel = lp, lc
                            done_reason = "loop"
                            if hasattr(resp, "close"):
                                resp.close()
                            break
            else:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
                if "error" in data and data["error"]:
                    raise RuntimeError(f"Ollama error: {data['error']}")
                model_returned = data.get("model", served_model)
                msg = data.get("message") or {}
                raw_content_parts.append(msg.get("content") or "")
                thinking_parts.append(msg.get("thinking") or "")
                prompt_tok = int(data.get("prompt_eval_count") or 0)
                compl_tok = int(data.get("eval_count") or 0)
                prompt_eval_ns = int(data.get("prompt_eval_duration") or 0)
                eval_ns = int(data.get("eval_duration") or 0)
                done_reason = data.get("done_reason") or "stop"

        raw_elapsed = time.time() - t0
        latency = round(raw_elapsed, 3)
        raw_content = "".join(raw_content_parts)
        thinking = "".join(thinking_parts)
        if loop_pattern is None and done_reason != "loop":
            lp, lc = _check_loop(thinking, raw_content)
            if lp:
                loop_pattern, loop_channel = lp, lc
                done_reason = "loop"
        if loop_pattern is not None:
            finish_reason = "loop"
            content = ""
            had_inline_think = bool(re.search(r"<think>|<\|channel", raw_content, flags=re.IGNORECASE))
            if compl_tok == 0:
                compl_tok = max(streamed_chunks, (len(raw_content) + len(thinking)) // 4)
        else:
            finish_reason = "length" if done_reason == "length" else "stop"
            content, had_inline_think = strip_inline_think(raw_content)
        eval_sec = round(eval_ns / 1e9, 3) if eval_ns > 0 else None
        prompt_eval_sec = round(prompt_eval_ns / 1e9, 3) if prompt_eval_ns > 0 else None
        decode_tps = _tps(compl_tok, eval_ns / 1e9) if eval_ns > 0 else None
        return {
            "content": content,
            "model_requested": model_id,
            "model_returned": model_returned,
            "provider": "local:ollama-chat",
            "reasoning_effort": None,
            "temperature": temperature,
            "prompt_tokens": prompt_tok,
            "completion_tokens": compl_tok,
            "reasoning_tokens": 0,
            "total_tokens": prompt_tok + compl_tok,
            "cost_usd": 0.0,
            "latency_seconds": latency,
            "tokens_per_second": _tps(compl_tok, max(raw_elapsed, 1e-6)),
            **({"decode_tokens_per_second": decode_tps} if decode_tps is not None else {}),
            **({"prompt_eval_seconds": prompt_eval_sec} if prompt_eval_sec is not None else {}),
            **({"eval_seconds": eval_sec} if eval_sec is not None else {}),
            "finish_reason": finish_reason,
            "reasoned": bool(thinking or had_inline_think),
            **({"loop_detected": True, "loop_pattern": loop_pattern, "loop_channel": loop_channel} if loop_pattern else {}),
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
    if telemetry.get("error") or response.startswith("[ERROR:"):
        return False
    if telemetry.get("finish_reason") == "loop":
        return True
    return bool(response.strip() and telemetry.get("finish_reason") != "length")


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
    loops_so_far = 0
    ct_so_far = 0
    lat_so_far = 0.0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_to_idx = {ex.submit(_exec_one, tasks[i]): i for i in range(len(tasks))}
        for fut in concurrent.futures.as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            row = fut.result()
            results_list[idx] = row
            done_count += 1
            tel = row.get("telemetry") or {}
            if tel.get("finish_reason") == "loop":
                loops_so_far += 1
            ct_so_far += int(tel.get("completion_tokens") or 0)
            lat_so_far += float(tel.get("latency_seconds") or 0.0)
            if done_count % 15 == 0 or done_count == len(tasks):
                avg_tps = f", avg_tps={ct_so_far / lat_so_far:.1f}" if lat_so_far > 0 and ct_so_far > 0 else ""
                print(f"  [{model_id}] completed {done_count}/{len(tasks)} calls ({time.time() - t_start:.1f}s, loops={loops_so_far}{avg_tps})")

    results_list = [{k: v for k, v in r.items() if k != "grading"} for r in results_list]
    qa_responses = [r for r in results_list if r["task_type"] == "qa"]
    summary_responses = [r for r in results_list if r["task_type"] == "summary"]
    total_tokens = sum(r["telemetry"].get("total_tokens", 0) for r in results_list)
    total_completion_tokens = sum(r["telemetry"].get("completion_tokens", 0) for r in results_list)
    total_completion_time = round(sum(float(r["telemetry"].get("latency_seconds") or 0.0) for r in results_list), 2)
    decode_tps_vals = sorted(
        float(r["telemetry"]["decode_tokens_per_second"])
        for r in results_list
        if isinstance(r["telemetry"].get("decode_tokens_per_second"), (int, float))
    )
    median_decode_tps = round(decode_tps_vals[len(decode_tps_vals) // 2], 2) if decode_tps_vals else None
    loop_count = sum(1 for r in results_list if r["telemetry"].get("finish_reason") == "loop")

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
            "total_completion_tokens": total_completion_tokens,
            "total_cost_usd": 0.0,
            "total_completion_time_seconds": total_completion_time,
            "wall_time_seconds": round(time.time() - t_start, 2),
            "effective_tokens_per_second": _tps(total_completion_tokens, total_completion_time),
            **({"median_decode_tokens_per_second": median_decode_tps} if median_decode_tps is not None else {}),
            "loop_count": loop_count,
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
