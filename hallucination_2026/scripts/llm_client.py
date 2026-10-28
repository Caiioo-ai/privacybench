"""
Unified LLM API client, text normalizer, and hash-verified article loader
for PrivacyBench v2 (`hallucination_2026`).

Every LLM call records:
- model_id (requested and returned)
- provider (serving provider from OpenRouter or direct API)
- reasoning_effort
- temperature
- prompt_tokens, completion_tokens, reasoning_tokens, total_tokens
- cost_usd
- latency_seconds
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cometapi_client

SCRIPT_DIR = Path(__file__).resolve().parent
TASK_ROOT = SCRIPT_DIR.parent
REPO_ROOT = TASK_ROOT.parent


def _load_env_files() -> None:
    """Load .env from repo root or user home, with fallback if python-dotenv is absent."""
    env_candidates = [REPO_ROOT / ".env", TASK_ROOT / ".env", Path.home() / ".env"]
    try:
        from dotenv import load_dotenv
        for p in env_candidates:
            if p.exists():
                load_dotenv(dotenv_path=p, override=True)
    except ImportError:
        for p in env_candidates:
            if p.exists():
                for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    if s.startswith("export "):
                        s = s[7:].strip()
                    if "=" in s:
                        k, v = s.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'").strip('"')
                        os.environ[k] = v


_load_env_files()

# Fallback per-token pricing (USD per token) if OpenRouter response does not embed usage.cost
MODEL_PRICING_PER_TOKEN: Dict[str, Tuple[float, float]] = {
    "google/gemini-4-argon": (2.0e-6, 10.0e-6),
    "meta/muse-wonder": (4.0e-6, 20.0e-6),
    "x-ai/grok-4.7": (2.0e-6, 6.0e-6),
    "anthropic/claude-sonnet-5.5": (2.0e-6, 10.0e-6),
    "anthropic/claude-opus-5.5": (4.0e-6, 20.0e-6),
    "anthropic/claude-haiku-5.5": (0.1e-6, 0.5e-6),
    "anthropic/claude-sonnet-4.5": (3.0e-6, 15.0e-6),
    "anthropic/claude-haiku-4.5": (1.0e-6, 5.0e-6),
    "openai/gpt-6.1-sol": (2.0e-6, 10.0e-6),
    "openai/gpt-6-luna": (0.1e-6, 0.5e-6),
    "openai/gpt-4.1": (2.0e-6, 8.0e-6),
    "openai/gpt-4.1-mini": (0.4e-6, 1.6e-6),
    "openai/gpt-4o": (2.5e-6, 10.0e-6),
    "openai/gpt-4o-mini": (0.15e-6, 0.6e-6),
    "openai/o4-mini": (1.1e-6, 4.4e-6),
    "google/gemini-2.5-pro": (1.25e-6, 10.0e-6),
    "google/gemini-2.5-flash": (0.3e-6, 2.5e-6),
    "deepseek/deepseek-v4-pro-0813": (0.66e-6, 1.98e-6),
    "deepseek/deepseek-v4.1-flash": (0.02376e-6, 0.72e-6),
    "deepseek/deepseek-chat-v3-0324": (0.29e-6, 1.14e-6),
    "deepseek/deepseek-chat": (0.32e-6, 0.89e-6),
    "mistralai/mistral-large-4-0": (0.68e-6, 2.09e-6),
    "mistralai/mistral-small-3.1-24b-instruct": (0.351e-6, 0.555e-6),
    "meta/muse-glimmer-30b": (0.3e-6, 1.2e-6),
    "meta-llama/llama-3.3-70b-instruct": (0.59e-6, 0.79e-6),
    "meta-llama/llama-4-maverick": (0.1875e-6, 0.6525e-6),
    "qwen/qwen3.8-flash": (0.15e-6, 0.47e-6),
    "qwen/qwen-2.5-7b-instruct": (0.10e-6, 0.20e-6),
    "qwen/qwen-2.5-72b-instruct": (0.36e-6, 0.40e-6),
    "qwen/qwen3-32b": (0.08e-6, 0.28e-6),
    "z-ai/glm-5.3": (0.12e-6, 4.80e-6),
    "xiaomi/mimo-v2.6-pro": (0.435e-6, 0.87e-6),
    "moonshotai/kimi-k3": (0.45e-6, 15.0e-6),
    "qwen/qwen3.8-max-0902": (2.0e-6, 6.0e-6),
}


def normalize_text(raw: str) -> str:
    """Canonical text normalization for deterministic SHA-256 hashing and offset indexing."""
    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in text.split("\n")]
    joined = "\n".join(lines)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()


def compute_sha256(norm_text: str) -> str:
    """Return hex SHA-256 digest of normalized UTF-8 article text."""
    return hashlib.sha256(norm_text.encode("utf-8")).hexdigest()


def get_article_text(article: Dict[str, Any], verify_hash: bool = True) -> str:
    """
    Resolve article text from `article` record.
    - If `text_committed` is True, reads `norm_text` and verifies `sha256_normalized`.
    - If `text_committed` is False, fetches `url` (or `archive_url`), normalizes,
      and verifies `sha256_normalized`, failing loudly if the hash differs.
    """
    expected_hash = article.get("sha256_normalized")
    if article.get("text_committed") and article.get("norm_text"):
        norm = normalize_text(article["norm_text"])
        if verify_hash and expected_hash:
            actual_hash = compute_sha256(norm)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Article {article.get('id')} committed text SHA-256 mismatch: "
                    f"expected {expected_hash}, got {actual_hash}"
                )
        return norm

    # Fetch at runtime for uncommitted (copyrighted) sources
    urls = [u for u in [article.get("url"), article.get("archive_url")] if u]
    if not urls:
        raise RuntimeError(
            f"Article {article.get('id')} has text_committed=False and no URL to fetch."
        )
    last_err = None
    raw_fetched = None
    for url in urls:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "PrivacyBench-Eval/2.0 (https://github.com/Caiioo-ai/privacybench)"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw_fetched = resp.read().decode("utf-8", errors="replace")
            break
        except Exception as e:
            last_err = e

    if raw_fetched is None:
        raise RuntimeError(
            f"Failed to fetch uncommitted article {article.get('id')} from {urls}: {last_err}"
        )

    norm = normalize_text(raw_fetched)
    actual_hash = compute_sha256(norm)
    if verify_hash and expected_hash and actual_hash != expected_hash:
        raise RuntimeError(
            f"Hash verification FAILED for runtime-fetched article {article.get('id')} "
            f"({article.get('url')}): expected SHA-256 {expected_hash}, got {actual_hash}. "
            f"Source text may have been modified or paywalled."
        )
    return norm


def is_article_eligible_for_model(article: Dict[str, Any], model_meta: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Check whether an article was published strictly after a model's effective cutoff.
    Returns (is_eligible, reason_string).
    """
    pub_date = article.get("published_at", "")
    cutoff = model_meta.get("effective_cutoff") or model_meta.get("vendor_published_cutoff") or model_meta.get("public_release_date")
    cutoff_src = model_meta.get("cutoff_source", "unknown")
    if not pub_date or not cutoff:
        raise ValueError(f"Missing published_at ({pub_date!r}) or effective_cutoff ({cutoff!r})")
    eligible = pub_date > cutoff
    reason = (
        f"published_at={pub_date} > effective_cutoff={cutoff} ({cutoff_src})"
        if eligible
        else f"INELIGIBLE: published_at={pub_date} <= effective_cutoff={cutoff} ({cutoff_src})"
    )
    return eligible, reason


def call_llm(
    model_id: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 512,
    reasoning_effort: Optional[str] = None,
    response_format: Optional[Dict[str, Any]] = None,
    timeout: int = 120,
    max_retries: int = 4,
    cache_prefix: Optional[str] = None,
    cache_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute an LLM chat completion call via OpenRouter (or CometAPI after `use_cometapi`, or local Ollama if
    `ollama:` prefix) and return a structured telemetry dict containing content + full metadata.
    `cache_prefix` / `cache_key` mark the prompt's shared prefix for gateways that cache explicitly (CometAPI).
    """
    if model_id.startswith("ollama:"):
        ollama_model = model_id[len("ollama:"):]
        return _call_ollama(ollama_model, messages, temperature, max_tokens, timeout)
    if _GATEWAY["name"] == cometapi_client.GATEWAY:
        return _call_cometapi(model_id, messages, temperature, max_tokens, reasoning_effort, response_format,
                              timeout, max_retries, cache_prefix, cache_key)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in environment or .env.")

    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}
    if response_format is not None:
        payload["response_format"] = response_format

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Caiioo-ai/privacybench",
        "X-Title": "PrivacyBench-v2-Hallucination-2026",
    }

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw_body = resp.read().decode("utf-8", errors="replace")
            latency = round(time.time() - t0, 3)
            data = json.loads(raw_body)
            if "error" in data:
                raise RuntimeError(f"OpenRouter error payload: {data['error']}")

            choices = data.get("choices") or [{}]
            msg = choices[0].get("message") or {}
            content = (msg.get("content") or "").strip()
            finish_reason = choices[0].get("finish_reason")
            if not content and finish_reason == "length" and attempt < max_retries:
                payload["max_tokens"] = min(16384, int(payload["max_tokens"]) * 2)
                continue
            usage = data.get("usage") or {}
            prompt_tok = int(usage.get("prompt_tokens") or 0)
            compl_tok = int(usage.get("completion_tokens") or 0)
            total_tok = int(usage.get("total_tokens") or (prompt_tok + compl_tok))
            details = usage.get("completion_tokens_details") or {}
            reasoning_tok = int(details.get("reasoning_tokens") or 0)

            # Cost from OpenRouter usage.cost if present, else fallback pricing table
            cost_usd = usage.get("cost")
            if cost_usd is None:
                p_rate, c_rate = MODEL_PRICING_PER_TOKEN.get(model_id, (1.0e-6, 3.0e-6))
                cost_usd = prompt_tok * p_rate + compl_tok * c_rate
            cost_usd = round(float(cost_usd), 6)

            return {
                "content": content,
                "model_requested": model_id,
                "model_returned": data.get("model", model_id),
                "provider": data.get("provider", "OpenRouter"),
                "reasoning_effort": reasoning_effort,
                "temperature": temperature,
                "prompt_tokens": prompt_tok,
                "completion_tokens": compl_tok,
                "reasoning_tokens": reasoning_tok,
                "total_tokens": total_tok,
                "cost_usd": cost_usd,
                "latency_seconds": latency,
                "finish_reason": choices[0].get("finish_reason"),
            }
        except urllib.error.HTTPError as he:
            err_body = ""
            try:
                err_body = he.read().decode("utf-8", errors="ignore")[:300]
            except Exception:
                pass
            # Fail loud: a refused model (e.g. Meta's 403 user_blocked) is never replaced by another model. The
            # 79f9508-983c5c8 client swapped meta/muse-wonder for meta/muse-glimmer-30b here, so a blocked judge or
            # target was silently served by a different model under the requested name.
            last_error = RuntimeError(f"HTTP {he.code} for {model_id}: {err_body}")
            if he.code in (400, 401, 402, 403):
                break
            time.sleep(1.5 * attempt)
        except Exception as e:
            last_error = e
            time.sleep(1.5 * attempt)

    raise RuntimeError(f"LLM call failed after {max_retries} attempts for {model_id}: {last_error}")


# The gateway every call in this process goes through; OpenRouter unless a script switches to CometAPI.
_GATEWAY: Dict[str, Any] = {"name": "openrouter", "routes": {}}


def cometapi_routes(models_file: Path, model_ids: List[str], catalog: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Each model's registry route (`models.json` -> `cometapi`) with the CometAPI catalogue facts it stands on."""
    registry = {m["model_id"]: m for m in json.loads(Path(models_file).read_text(encoding="utf-8"))["models"]}
    routes: Dict[str, Dict[str, Any]] = {}
    for model_id in model_ids:
        route = (registry.get(model_id) or {}).get("cometapi")
        if not isinstance(route, dict):
            raise RuntimeError(f"{model_id}: no CometAPI route in the model registry")
        routes[model_id] = {**route, **cometapi_client.route_evidence(catalog, route)}
    return routes


def use_cometapi(routes: Dict[str, Dict[str, Any]]) -> None:
    """Send every later call through CometAPI; a model without a route here fails loud, never falls back."""
    _GATEWAY["name"] = cometapi_client.GATEWAY
    _GATEWAY["routes"] = dict(routes)


def gateway_name() -> str:
    return _GATEWAY["name"]


def loaded_route(model_id: str) -> Optional[Dict[str, Any]]:
    """The CometAPI route (with its catalogue evidence) this run loaded for `model_id`; None on OpenRouter."""
    if _GATEWAY["name"] != cometapi_client.GATEWAY:
        return None
    route = _GATEWAY["routes"].get(model_id)
    if route is None:
        raise RuntimeError(f"{model_id}: no CometAPI route was loaded for this run")
    return route


def _call_cometapi(model_id, messages, temperature, max_tokens, reasoning_effort, response_format, timeout,
                   max_retries, cache_prefix, cache_key) -> Dict[str, Any]:
    route = _GATEWAY["routes"].get(model_id)
    if route is None:
        raise RuntimeError(f"{model_id}: no CometAPI route was loaded for this run")
    if response_format is not None:
        raise RuntimeError("response_format is not carried on the CometAPI wires")
    if len(messages) != 1 or messages[0].get("role") != "user":
        raise RuntimeError("the CometAPI client sends one user prompt")
    last_error: Optional[Exception] = None
    attempt = 0
    while attempt < max_retries:
        gate = cometapi_client.QUOTA_GATE
        gate.before_call()
        try:
            response = cometapi_client.call(model_id, route, route["pricing"], messages[0]["content"],
                                            temperature=temperature, max_tokens=max_tokens,
                                            reasoning_effort=reasoning_effort, cache_prefix=cache_prefix,
                                            cache_key=cache_key, timeout=timeout)
            gate.on_success()
            return response
        except cometapi_client.CometApiError as e:
            if cometapi_client.is_quota_refusal(e):
                gate.on_quota()  # waits out the reload; not an attempt (raises QuotaExhausted past the cap)
                continue
            gate.release()
            attempt += 1
            last_error = e
            # A refused request fails the same way again; a call another channel served is retried, since CometAPI
            # spreads calls over channels and the next one usually lands on the requested model. A content filter
            # is a channel's (gpt-6.1-sol's Azure channel refused a judge prompt as invalid_prompt).
            if re.match(r"HTTP 4(00|01|02|03|04)\b", str(e)) and "invalid_prompt" not in str(e):
                break
        except Exception as e:
            gate.release()
            attempt += 1
            last_error = e
        time.sleep(1.5 * attempt)
    raise RuntimeError(f"LLM call failed after {max_retries} attempts for {model_id}: {last_error}")


def served_record(resp: Dict[str, Any]) -> Dict[str, Any]:
    """Telemetry of one call without its content: who was asked, who served it, effort, tokens, cost, latency."""
    return {k: v for k, v in resp.items() if k != "content"}


def _call_ollama(
    ollama_model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> Dict[str, Any]:
    prompt_parts = []
    for m in messages:
        role = m.get("role", "user")
        if role == "system":
            prompt_parts.append(f"System: {m['content']}")
        else:
            prompt_parts.append(m["content"])
    full_prompt = "\n\n".join(prompt_parts)
    payload = {
        "model": ollama_model,
        "prompt": full_prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    t0 = time.time()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="replace"))
    latency = round(time.time() - t0, 3)
    p_tok = int(data.get("prompt_eval_count") or 0)
    c_tok = int(data.get("eval_count") or 0)
    return {
        "content": (data.get("response") or "").strip(),
        "model_requested": f"ollama:{ollama_model}",
        "model_returned": data.get("model", ollama_model),
        "provider": "local-ollama",
        "reasoning_effort": None,
        "temperature": temperature,
        "prompt_tokens": p_tok,
        "completion_tokens": c_tok,
        "reasoning_tokens": 0,
        "total_tokens": p_tok + c_tok,
        "cost_usd": 0.0,
        "latency_seconds": latency,
        "finish_reason": "stop",
    }
