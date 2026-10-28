"""CometAPI transport for PrivacyBench v2 (`hallucination_2026`).

Each model's route is registry data (`models.json` -> `cometapi`: the CometAPI id, the wire, and the exact served ids
accepted as that model), checked against CometAPI's own catalogue before inference; nothing here infers a wire or an
identity from a model's name. Telemetry keeps the OpenRouter client's shape (prompt_tokens is the WHOLE prompt, cached
and written tokens are subsets) so the rollup, the ingest and repricing read both alike.

Caching (measured 2026-10-28, see tasks/agent-reports-2026-10-28/privacybench-top-unmeasured in contextflow):
- anthropic-messages: an ephemeral `cache_control` block ending exactly where the per-item text begins, plus a
  `session-id` header (CometAPI's sticky routing; without it no read ever hit). The split is token-identical to the
  single string (5/5 items, haiku-5.5, against its OpenRouter run).
- openai-responses: `prompt_cache_key`, `store: false`, and NO `session-id` (with one, a call was billed 7,268 input
  tokens for a 3,052-token prompt). Chat Completions never hit the cache for the same model, so OpenAI runs here.
- openai-chat / gemini: the provider's implicit caching only.

Cost: CometAPI bills ratio x its catalogue price (0.8 measured against the billing counter on three wires, to the
cent's ten-thousandth); it returns no cost, so the cost is computed from the same catalogue entry and recorded with
its basis.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

BASE_URL = 'https://api.cometapi.com'
MODELS_URL = f'{BASE_URL}/v1/models'
PRICING_URL = f'{BASE_URL}/api/models'
GATEWAY = 'cometapi'

# The words CometAPI's catalogue uses to declare each wire, across its live entry shapes (`supported_endpoint_types`
# on /v1/models, `endpoints` on /api/models).
WIRE_DECLARATIONS = {
    'anthropic-messages': {'anthropic', 'anthropic-messages'},
    'openai-responses': {'responses', 'openai-response'},
    'openai-chat': {'openai', 'chat', 'openai-chat'},
    'gemini': {'gemini'},
}
PRICE_VARIABLES = ('p', 'c', 'cr', 'cc', 'cc1h')


class CometApiError(RuntimeError):
    pass


def _get(url: str, key: str) -> Tuple[Any, bytes]:
    req = urllib.request.Request(url, headers={'Authorization': f'Bearer {key}'})
    with urllib.request.urlopen(req, timeout=60) as response:
        raw = response.read()
    return json.loads(raw), raw


def api_key() -> str:
    key = os.environ.get('COMETAPI_KEY')
    if not key:
        raise CometApiError('COMETAPI_KEY is not set in environment or .env.')
    return key


def fetch_catalog() -> dict:
    """CometAPI's model list (declared wires) and gateway catalogue (prices), with their digests as the receipt."""
    key = api_key()
    models, models_raw = _get(MODELS_URL, key)
    pricing, pricing_raw = _get(PRICING_URL, key)
    if not isinstance(models.get('data'), list) or not isinstance(pricing.get('data'), list):
        raise CometApiError('CometAPI catalogue is malformed')
    return {'models': models['data'], 'pricing': pricing['data'],
            '_retrieval': {'models_url': MODELS_URL, 'pricing_url': PRICING_URL,
                           'retrieved_at': datetime.now(timezone.utc).isoformat(),
                           'models_sha256': hashlib.sha256(models_raw).hexdigest(),
                           'pricing_sha256': hashlib.sha256(pricing_raw).hexdigest()}}


def _claimed_ids(entry: dict) -> Dict[str, Optional[dict]]:
    """Every id a gateway-catalogue entry prices, mapped to that id's own pricing when the entry gives one."""
    matched = entry.get('matched_models')
    claimed: Dict[str, Optional[dict]] = {}
    if isinstance(matched, dict):
        for model_id, own in matched.items():
            claimed[model_id] = own if isinstance(own, dict) else None
    elif isinstance(matched, list):
        for item in matched:
            if isinstance(item, str):
                claimed[item] = None
    for own_id in (entry.get('id'), entry.get('code')):
        if isinstance(own_id, str):
            claimed.setdefault(own_id, None)
    return claimed


# The path each wire posts to; an `endpoints` list may name paths instead of words.
WIRE_PATHS = {'anthropic-messages': '/v1/messages', 'openai-responses': '/v1/responses',
              'openai-chat': '/v1/chat/completions', 'gemini': '/v1beta'}


def _endpoint_words(endpoints: Any) -> set:
    """Words and paths an `/api/models` `endpoints` field declares, across its live shapes (as contextflow's
    `parseCometApiEndpoints`): null, a list of paths, a dict of word -> path | {path}, or either JSON-encoded."""
    if isinstance(endpoints, str):
        try:
            endpoints = json.loads(endpoints)
        except ValueError:
            return {endpoints.strip()} - {''}
    if isinstance(endpoints, list):
        return {e.strip() for e in endpoints if isinstance(e, str) and e.strip()}
    if isinstance(endpoints, dict):
        words = set(endpoints.keys())
        for value in endpoints.values():
            path = value if isinstance(value, str) else (value.get('path') if isinstance(value, dict) else None)
            if isinstance(path, str) and path.strip():
                words.add(path.strip())
        return words
    return set()


def _declares(words: set, wire: str) -> bool:
    return bool(words & WIRE_DECLARATIONS[wire]) or any(w.startswith(WIRE_PATHS[wire]) for w in words)


def route_evidence(catalog: dict, route: dict) -> dict:
    """The catalogue facts a route stands on; raises unless the id is listed once, priced once and declares the wire."""
    model = route.get('model')
    wire = route.get('wire')
    if wire not in WIRE_DECLARATIONS:
        raise CometApiError(f'{model}: unknown wire {wire!r}')
    listed = [m for m in catalog['models'] if isinstance(m, dict) and m.get('id') == model]
    if len(listed) != 1:
        raise CometApiError(f'{model}: expected one CometAPI model listing, found {len(listed)}')
    priced = []
    words = set(listed[0].get('supported_endpoint_types') or [])
    for entry in catalog['pricing']:
        if not isinstance(entry, dict):
            continue
        claimed = _claimed_ids(entry)
        if model in claimed:
            priced.append(claimed[model] or entry.get('pricing'))
            words |= _endpoint_words(entry.get('endpoints'))
    distinct = {json.dumps(p, sort_keys=True) for p in priced}
    if len(distinct) != 1:
        raise CometApiError(f'{model}: expected one CometAPI price, found {len(distinct)}')
    if not _declares(words, wire):
        raise CometApiError(f'{model}: CometAPI does not declare the {wire} wire (declares {sorted(words)})')
    return {'model': model, 'wire': wire, 'declared_endpoints': sorted(words), 'pricing': json.loads(distinct.pop())}


# ── Cost ─────────────────────────────────────────────────────────────────────────────────────────────────────────

_TIERED = re.compile(r'^len\s*(<=|<)\s*(\d+)\s*\?\s*tier\("([\w-]+)",\s*(.+?)\)\s*:\s*tier\("([\w-]+)",\s*(.+?)\)$')
_TERM = re.compile(r'^(\w+)\s*\*\s*([0-9.]+)$')


def _rates(sum_expression: str) -> Dict[str, float]:
    rates: Dict[str, float] = {}
    for term in sum_expression.split('+'):
        match = _TERM.match(term.strip())
        if not match or match.group(1) not in PRICE_VARIABLES or match.group(1) in rates:
            raise CometApiError(f'unreadable price term {term.strip()!r}')
        rates[match.group(1)] = float(match.group(2))
    return rates


def price_rates(pricing: dict, prompt_tokens: int) -> Tuple[Dict[str, float], dict]:
    """Per-million rates for each billed token kind at this prompt length, and the basis they were read from."""
    if not isinstance(pricing, dict) or pricing.get('currency') != 'USD / M Tokens':
        raise CometApiError(f'unsupported CometAPI pricing {pricing!r}')
    if pricing.get('per_request') or pricing.get('per_second'):
        raise CometApiError('per-request or per-second pricing is not token pricing')
    ratio = pricing.get('ratio')
    if not isinstance(ratio, (int, float)) or not 0 < ratio <= 1:
        raise CometApiError(f'unreadable CometAPI ratio {ratio!r}')
    if pricing.get('billing_mode') == 'tiered_expr':
        match = _TIERED.match(str(pricing.get('billing_expr', '')).strip())
        if not match:
            raise CometApiError(f'unreadable billing_expr {pricing.get("billing_expr")!r}')
        comparator, limit, short_name, short_sum, long_name, long_sum = match.groups()
        short = prompt_tokens <= int(limit) if comparator == '<=' else prompt_tokens < int(limit)
        tier, expression = (short_name, short_sum) if short else (long_name, long_sum)
        rates = _rates(expression)
        basis = {'mode': 'tiered_expr', 'billing_expr': pricing['billing_expr'], 'tier': tier, 'ratio': ratio}
    else:
        if not isinstance(pricing.get('input'), (int, float)) or not isinstance(pricing.get('output'), (int, float)):
            raise CometApiError(f'pricing without input/output rates {pricing!r}')
        rates = {'p': float(pricing['input']), 'c': float(pricing['output'])}
        assumptions = []
        if isinstance(pricing.get('cached_input'), (int, float)):
            rates['cr'] = float(pricing['cached_input'])
        else:
            rates['cr'] = rates['p']
            assumptions.append('cache-read-billed-at-input-rate')
        if isinstance(pricing.get('cache_write'), (int, float)):
            rates['cc'] = float(pricing['cache_write'])
        else:
            rates['cc'] = rates['p']
            assumptions.append('cache-write-billed-at-input-rate')
        basis = {'mode': 'flat', 'input': pricing['input'], 'output': pricing['output'],
                 'cached_input': pricing.get('cached_input'), 'cache_write': pricing.get('cache_write'),
                 'ratio': ratio, 'assumptions': assumptions}
    return rates, basis


def cost_usd(pricing: dict, usage: dict) -> Tuple[float, dict]:
    """ratio x catalogue price of one call's tokens; uncached prompt = whole prompt - cache reads - cache writes."""
    cached, written = usage['cached_tokens'], usage['cache_write_tokens']
    uncached = usage['prompt_tokens'] - cached - written
    if uncached < 0:
        raise CometApiError(f'cached + written tokens exceed the prompt: {usage}')
    rates, basis = price_rates(pricing, usage['prompt_tokens'])
    counts = {'p': uncached, 'c': usage['completion_tokens'], 'cr': cached, 'cc': written, 'cc1h': 0}
    for kind, count in counts.items():
        if count and kind not in rates:
            raise CometApiError(f'{count} {kind} tokens billed, but the price names no {kind} rate')
    list_usd = sum(rates.get(kind, 0.0) * count for kind, count in counts.items()) / 1e6
    return round(list_usd * basis['ratio'], 8), basis


# ── Wires ────────────────────────────────────────────────────────────────────────────────────────────────────────

def _split(prompt: str, cache_prefix: Optional[str]) -> Tuple[str, str]:
    if cache_prefix is None:
        return prompt, ''
    if not cache_prefix or not prompt.startswith(cache_prefix) or len(cache_prefix) == len(prompt):
        raise CometApiError('cache_prefix must be a proper, non-empty prefix of the prompt')
    return cache_prefix, prompt[len(cache_prefix):]


def build_request(wire: str, model: str, prompt: str, *, temperature: float, max_tokens: int,
                  reasoning_effort: Optional[str], cache_prefix: Optional[str], cache_key: Optional[str]
                  ) -> Tuple[str, dict, dict, dict]:
    """(path, body, extra headers, wire record) for one single-turn user prompt. The record says what was sent."""
    record: Dict[str, Any] = {'wire': wire, 'temperature_sent': temperature, 'cache': 'implicit'}
    headers: Dict[str, str] = {}
    if wire == 'anthropic-messages':
        prefix, rest = _split(prompt, cache_prefix)
        content: List[dict] = ([{'type': 'text', 'text': prefix, 'cache_control': {'type': 'ephemeral'}},
                                {'type': 'text', 'text': rest}] if cache_prefix is not None
                               else [{'type': 'text', 'text': prompt}])
        body: Dict[str, Any] = {'model': model, 'max_tokens': max_tokens,
                                'messages': [{'role': 'user', 'content': content}]}
        if reasoning_effort is not None:
            # Anthropic accepts no temperature but 1 beside thinking; the planner's wire (contextflow anthropic-wire.ts).
            body['thinking'] = {'type': 'adaptive'}
            body['output_config'] = {'effort': reasoning_effort}
            record['temperature_sent'] = None
        else:
            body['temperature'] = temperature
        headers.update({'anthropic-version': '2023-06-01', 'anthropic-beta': 'prompt-caching-2024-07-31'})
        if cache_key:
            headers['session-id'] = cache_key
        if cache_prefix is not None:
            record['cache'] = 'cache_control+session-id'
        return '/v1/messages', body, headers, record
    if wire == 'openai-responses':
        body = {'model': model, 'input': prompt, 'max_output_tokens': max_tokens, 'store': False}
        if reasoning_effort is not None:
            # OpenAI's reasoning models refuse a temperature (OpenRouter drops it silently).
            body['reasoning'] = {'effort': reasoning_effort}
            record['temperature_sent'] = None
        else:
            body['temperature'] = temperature
        if cache_key:
            body['prompt_cache_key'] = cache_key
            record['cache'] = 'prompt_cache_key'
        return '/v1/responses', body, headers, record
    if wire == 'openai-chat':
        body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': max_tokens,
                'temperature': temperature}
        if reasoning_effort is not None:
            body['reasoning_effort'] = reasoning_effort
        return '/v1/chat/completions', body, headers, record
    if wire == 'gemini':
        config: Dict[str, Any] = {'temperature': temperature, 'maxOutputTokens': max_tokens}
        if reasoning_effort is not None:
            config['thinkingConfig'] = {'thinkingLevel': reasoning_effort}
        body = {'contents': [{'role': 'user', 'parts': [{'text': prompt}]}], 'generationConfig': config}
        headers['x-goog-api-key'] = api_key()
        return f'/v1beta/models/{model}:generateContent', body, headers, record
    raise CometApiError(f'unknown wire {wire!r}')


def _int(value: Any) -> int:
    if value is None:
        return 0
    if not isinstance(value, int) or value < 0:
        raise CometApiError(f'unreadable token count {value!r}')
    return value


def parse_response(wire: str, data: dict) -> dict:
    """Text, finish reason, served id and usage in the OpenRouter telemetry shape (prompt = whole prompt)."""
    if wire == 'anthropic-messages':
        u = data.get('usage') or {}
        fresh, written, read = _int(u.get('input_tokens')), _int(u.get('cache_creation_input_tokens')), _int(u.get('cache_read_input_tokens'))
        text = ''.join(b.get('text', '') for b in data.get('content') or [] if isinstance(b, dict) and b.get('type') == 'text')
        finish = {'end_turn': 'stop', 'stop_sequence': 'stop', 'max_tokens': 'length'}.get(data.get('stop_reason'), data.get('stop_reason'))
        usage = {'prompt_tokens': fresh + written + read, 'completion_tokens': _int(u.get('output_tokens')),
                 'reasoning_tokens': _int((u.get('output_tokens_details') or {}).get('thinking_tokens')),
                 'cached_tokens': read, 'cache_write_tokens': written}
        return {'content': text, 'finish_reason': finish, 'served': data.get('model'), 'usage': usage}
    if wire == 'openai-responses':
        u = data.get('usage') or {}
        details = u.get('input_tokens_details') or {}
        text = ''.join(c.get('text', '') for o in data.get('output') or [] if isinstance(o, dict) and o.get('type') == 'message'
                       for c in o.get('content') or [] if isinstance(c, dict) and c.get('type') == 'output_text')
        incomplete = (data.get('incomplete_details') or {}).get('reason')
        finish = 'stop' if data.get('status') == 'completed' else ('length' if incomplete == 'max_output_tokens' else data.get('status'))
        usage = {'prompt_tokens': _int(u.get('input_tokens')), 'completion_tokens': _int(u.get('output_tokens')),
                 'reasoning_tokens': _int((u.get('output_tokens_details') or {}).get('reasoning_tokens')),
                 'cached_tokens': _int(details.get('cached_tokens')), 'cache_write_tokens': _int(details.get('cache_write_tokens'))}
        return {'content': text, 'finish_reason': finish, 'served': data.get('model'), 'usage': usage}
    if wire == 'openai-chat':
        u = data.get('usage') or {}
        choice = (data.get('choices') or [{}])[0]
        details = u.get('prompt_tokens_details') or {}
        usage = {'prompt_tokens': _int(u.get('prompt_tokens')), 'completion_tokens': _int(u.get('completion_tokens')),
                 'reasoning_tokens': _int((u.get('completion_tokens_details') or {}).get('reasoning_tokens')),
                 'cached_tokens': _int(details.get('cached_tokens')), 'cache_write_tokens': _int(details.get('cache_write_tokens'))}
        return {'content': ((choice.get('message') or {}).get('content') or ''), 'finish_reason': choice.get('finish_reason'),
                'served': data.get('model'), 'usage': usage}
    if wire == 'gemini':
        u = data.get('usageMetadata') or {}
        candidate = (data.get('candidates') or [{}])[0]
        parts = (candidate.get('content') or {}).get('parts') or []
        text = ''.join(p.get('text', '') for p in parts if isinstance(p, dict) and not p.get('thought'))
        finish = {'STOP': 'stop', 'MAX_TOKENS': 'length'}.get(candidate.get('finishReason'), candidate.get('finishReason'))
        thoughts = _int(u.get('thoughtsTokenCount'))
        usage = {'prompt_tokens': _int(u.get('promptTokenCount')), 'completion_tokens': _int(u.get('candidatesTokenCount')) + thoughts,
                 'reasoning_tokens': thoughts, 'cached_tokens': _int(u.get('cachedContentTokenCount')), 'cache_write_tokens': 0}
        return {'content': text, 'finish_reason': finish, 'served': data.get('modelVersion'), 'usage': usage}
    raise CometApiError(f'unknown wire {wire!r}')


def served_as(route: dict, served: Any, model_id: str) -> str:
    """The measured model's id when CometAPI says the requested model (or an id the registry accepts as it) served."""
    if served == route['model'] or served in (route.get('served_ids') or []):
        return model_id
    raise CometApiError(f'{model_id}: CometAPI says {served!r} served the call, not {route["model"]!r} '
                        f'or an accepted served id {route.get("served_ids") or []}')


def call(model_id: str, route: dict, pricing: dict, prompt: str, *, temperature: float, max_tokens: int,
         reasoning_effort: Optional[str], cache_prefix: Optional[str] = None, cache_key: Optional[str] = None,
         timeout: int = 120) -> Dict[str, Any]:
    """One single-turn call; the telemetry dict the OpenRouter client returns, plus the gateway's receipts."""
    key = api_key()
    path, body, headers, record = build_request(route['wire'], route['model'], prompt, temperature=temperature,
                                                max_tokens=max_tokens, reasoning_effort=reasoning_effort,
                                                cache_prefix=cache_prefix, cache_key=cache_key)
    request = urllib.request.Request(BASE_URL + path, data=json.dumps(body).encode('utf-8'), method='POST',
                                     headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json',
                                              **headers})
    t0 = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode('utf-8', errors='replace')
    except urllib.error.HTTPError as he:
        detail = he.read().decode('utf-8', errors='ignore')[:300]
        raise CometApiError(f'HTTP {he.code} for {model_id} via CometAPI {route["model"]}: {detail}') from None
    latency = round(time.time() - t0, 3)
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get('error'):
        raise CometApiError(f'CometAPI error payload for {model_id}: {str(data)[:300]}')
    parsed = parse_response(route['wire'], data)
    usage = parsed['usage']
    cost, basis = cost_usd(pricing, usage)
    return {
        'content': parsed['content'].strip(),
        'model_requested': model_id,
        'model_returned': served_as(route, parsed['served'], model_id),
        'provider': GATEWAY,
        'reasoning_effort': reasoning_effort,
        'temperature': temperature,
        'prompt_tokens': usage['prompt_tokens'],
        'completion_tokens': usage['completion_tokens'],
        'reasoning_tokens': usage['reasoning_tokens'],
        'total_tokens': usage['prompt_tokens'] + usage['completion_tokens'],
        'cost_usd': cost,
        'latency_seconds': latency,
        'finish_reason': parsed['finish_reason'],
        'gateway': GATEWAY,
        'gateway_model': route['model'],
        'provider_model_returned': parsed['served'],
        'cached_tokens': usage['cached_tokens'],
        'cache_write_tokens': usage['cache_write_tokens'],
        'cost_source': 'cometapi-catalogue-ratio',
        'cost_basis': basis,
        **record,
    }
