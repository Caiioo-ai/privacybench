"""Highest declared native effort for PrivacyBench's OpenRouter API condition.

OpenRouter orders supported_efforts highest first. Null is gateway-any, not
proof of a native ceiling; omitted selection is explicitly not applicable.
https://github.com/OpenRouterTeam/terraform-provider-openrouter/blob/main/docs/data-sources/model.md
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import urllib.request

CATALOG_URL = 'https://openrouter.ai/api/v1/models'
POLICY_VERSION = 'highest-declared-nondelegating-api-v1'
# Semantics, never a ranking: the catalogue alone supplies effort order.
# Unknown words need route-specific evidence before selection or exclusion.
# Codex describes its ultra as delegating; that does not establish API semantics.
EFFORT_SEMANTICS = {**{word: 'native' for word in ('max', 'xhigh', 'high', 'medium', 'low', 'minimal')},
                    'none': 'off'}
POLICY = {'version': POLICY_VERSION, 'ordering': 'openrouter-declared-descending',
          'semantics': EFFORT_SEMANTICS, 'gateway_any': 'unresolved',
          'no_selection': 'not-applicable', 'unknown': 'fail-before-inference'}


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class EffortPolicyError(ValueError):
    pass


def fetch_capability_catalog() -> dict:
    """Public unauthenticated GET only; no completion endpoint or credentials."""
    with urllib.request.urlopen(CATALOG_URL, timeout=30) as response:
        raw = response.read()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get('data'), list):
        raise EffortPolicyError('OpenRouter capability catalogue is malformed')
    return {**payload, '_retrieval': {'url': CATALOG_URL,
            'retrieved_at': datetime.now(timezone.utc).isoformat(),
            'response_sha256': hashlib.sha256(raw).hexdigest()}}


def _declaration(model_id: str, catalog: dict) -> tuple:
    """The model's one exact catalogue row: its evidence record, reasoning declaration and supported parameters."""
    if not isinstance(catalog, dict) or not isinstance(catalog.get('data'), list):
        raise EffortPolicyError('Missing/malformed OpenRouter capability catalogue')
    rows = [r for r in catalog['data'] if isinstance(r, dict) and r.get('id') == model_id]
    if len(rows) != 1:
        raise EffortPolicyError(f'{model_id}: expected one exact public capability declaration; found {len(rows)}')
    row = rows[0]
    params = row.get('supported_parameters')
    if not isinstance(params, list) or any(not isinstance(p, str) or not p for p in params):
        raise EffortPolicyError(f'{model_id}: malformed/missing supported_parameters')
    reasoning = row.get('reasoning')
    evidence = {'model_id': model_id, 'reasoning_present': 'reasoning' in row,
                'reasoning': reasoning, 'supported_parameters': params,
                'alias_target': row.get('alias_target')}
    if row.get('alias_target') is not None:
        raise EffortPolicyError(f'{model_id}: alias target is not an exact native model declaration')
    return evidence, reasoning, params


def resolve_effort(model_id: str, catalog: dict) -> dict:
    evidence, reasoning, params = _declaration(model_id, catalog)
    state, effort = 'not-applicable-no-selection', None
    if reasoning is None:
        if any('reasoning' in p or 'thinking' in p for p in params):
            raise EffortPolicyError(f'{model_id}: reasoning capability unknown; native maximum unestablished')
    elif not isinstance(reasoning, dict):
        raise EffortPolicyError(f'{model_id}: malformed reasoning declaration')
    else:
        for key in ('mandatory', 'default_enabled', 'supports_max_tokens'):
            if key in reasoning and reasoning[key] is not None and type(reasoning[key]) is not bool:
                raise EffortPolicyError(f'{model_id}: malformed {key}')
        default = reasoning.get('default_effort')
        if default is not None and (not isinstance(default, str) or not default.strip()):
            raise EffortPolicyError(f'{model_id}: malformed default_effort')
        if 'supported_efforts' not in reasoning:
            if 'reasoning_effort' in params or default is not None:
                raise EffortPolicyError(f'{model_id}: conflicting no-selection declaration')
            state = ('not-applicable-budget-only' if reasoning.get('supports_max_tokens') is True
                     else 'not-applicable-no-selection')
        else:
            levels = reasoning['supported_efforts']
            if levels is None:
                raise EffortPolicyError(f'{model_id}: explicit null accepts gateway-any effort; native maximum unestablished')
            if (not isinstance(levels, list) or not levels or
                    any(not isinstance(v, str) or not v or v != v.strip() for v in levels) or
                    len(set(levels)) != len(levels)):
                raise EffortPolicyError(f'{model_id}: malformed/empty supported_efforts')
            if 'reasoning_effort' not in params:
                raise EffortPolicyError(f'{model_id}: effort list conflicts with supported_parameters')
            if any(v not in EFFORT_SEMANTICS for v in levels):
                raise EffortPolicyError(f'{model_id}: unknown effort semantics; native maximum unestablished')
            if default is not None and default not in levels:
                raise EffortPolicyError(f'{model_id}: default_effort conflicts with supported_efforts')
            effort = next((v for v in levels if EFFORT_SEMANTICS[v] == 'native'), None)
            if effort is None:
                raise EffortPolicyError(f'{model_id}: no declared non-off, non-delegating native effort')
            state = 'selected-highest-native'
    identity = {'policy': POLICY, 'capability': evidence, 'state': state, 'selected_effort': effort}
    return {**identity, 'fingerprint': fingerprint(identity),
            'catalog_retrieval': catalog.get('_retrieval', {'url': CATALOG_URL, 'retrieved_at': None,
                                                          'response_sha256': None})}


# ── A requested level, snapped as the Caiioo router snaps it ─────────────────────────────────────────────────────
# Port of contextflow src/shared/reasoning-effort/levels.ts (`effortLadder`, `snapLevel`); the parity fixture
# tests/fixtures/snap_level_ts.json is generated from that source. The router ranks a turn at its level and each
# adapter snaps it onto the model's declared levels, so a run is comparable only at the rung the router would send.
LEVEL_POLICY_VERSION = 'requested-level-snapped-api-v1'
OFF_LEVEL = 'none'
LEVEL_POLICY = {'version': LEVEL_POLICY_VERSION, 'ordering': 'openrouter-declared-descending-merged',
                'snap': 'nearest-ladder-position-ties-lower', 'semantics': EFFORT_SEMANTICS,
                'gateway_any': 'unresolved', 'no_selection': 'not-applicable', 'unknown': 'fail-before-inference'}


def effort_ladder(descending_lists) -> list:
    """Every effort word the lists order, highest first: their merged order (first seen breaks ties and cycles)."""
    first_seen, below, indegree = {}, {}, {}

    def note(word):
        if word not in first_seen:
            first_seen[word] = len(first_seen)
            below[word] = set()
            indegree[word] = 0

    for levels in descending_lists:
        for i, word in enumerate(levels):
            note(word)
            if i == 0:
                continue
            upper = levels[i - 1]
            if upper == word or word in below[upper]:
                continue
            below[upper].add(word)
            indegree[word] += 1
    order, placed = [], set()
    while len(placed) < len(first_seen):
        ready = sorted((w for w in first_seen if w not in placed and indegree[w] == 0), key=first_seen.get)
        # A cycle leaves nothing ready: take the earliest-seen remaining word and carry on.
        nxt = ready[0] if ready else sorted((w for w in first_seen if w not in placed), key=first_seen.get)[0]
        placed.add(nxt)
        order.append(nxt)
        for lower in below[nxt]:
            indegree[lower] -= 1
    return order


def snap_level(requested: str, levels, ladder) -> str | None:
    """The declared level nearest `requested` on the ladder, ties toward the lower; never the off value."""
    if requested != OFF_LEVEL and requested in levels:
        return requested
    if requested not in ladder:
        return None
    want = ladder.index(requested)
    best, best_distance, best_index = None, float('inf'), -1
    for level in (w for w in ladder if w != OFF_LEVEL and w in levels):
        index = ladder.index(level)
        distance = abs(index - want)
        if distance < best_distance or (distance == best_distance and index > best_index):
            best, best_distance, best_index = level, distance, index
    return best


def resolve_level_effort(model_id: str, catalog: dict, requested: str) -> dict:
    """The effort a run at `requested` sends this model: the level itself, its snap, or none where none is selectable."""
    if requested not in EFFORT_SEMANTICS or EFFORT_SEMANTICS[requested] != 'native':
        raise EffortPolicyError(f'{requested!r} is not a native effort level')
    evidence, reasoning, params = _declaration(model_id, catalog)
    declared = [r['reasoning']['supported_efforts'] for r in catalog['data']
                if isinstance(r, dict) and isinstance(r.get('reasoning'), dict)
                and isinstance(r['reasoning'].get('supported_efforts'), list) and r['reasoning']['supported_efforts']]
    ladder = effort_ladder(declared)
    state, effort = 'not-applicable-no-selection', None
    if reasoning is None:
        if any('reasoning' in p or 'thinking' in p for p in params):
            raise EffortPolicyError(f'{model_id}: reasoning capability unknown; level unestablished')
    elif not isinstance(reasoning, dict):
        raise EffortPolicyError(f'{model_id}: malformed reasoning declaration')
    elif 'supported_efforts' not in reasoning:
        if 'reasoning_effort' in params or reasoning.get('default_effort') is not None:
            raise EffortPolicyError(f'{model_id}: conflicting no-selection declaration')
        state = ('not-applicable-budget-only' if reasoning.get('supports_max_tokens') is True
                 else 'not-applicable-no-selection')
    else:
        levels = reasoning['supported_efforts']
        if levels is None:
            raise EffortPolicyError(f'{model_id}: explicit null accepts gateway-any effort; level unestablished')
        if (not isinstance(levels, list) or not levels or
                any(not isinstance(v, str) or not v or v != v.strip() for v in levels) or len(set(levels)) != len(levels)):
            raise EffortPolicyError(f'{model_id}: malformed/empty supported_efforts')
        if 'reasoning_effort' not in params:
            raise EffortPolicyError(f'{model_id}: effort list conflicts with supported_parameters')
        if any(v not in EFFORT_SEMANTICS for v in levels):
            raise EffortPolicyError(f'{model_id}: unknown effort semantics; level unestablished')
        effort = snap_level(requested, levels, ladder)
        if effort is None or EFFORT_SEMANTICS.get(effort) != 'native':
            raise EffortPolicyError(f'{model_id}: no declared native level near {requested!r}')
        state = 'selected-requested-level' if effort == requested else 'selected-snapped-level'
    identity = {'policy': {**LEVEL_POLICY, 'requested_level': requested}, 'capability': evidence,
                'ladder': ladder, 'state': state, 'selected_effort': effort}
    return {**identity, 'fingerprint': fingerprint(identity),
            'catalog_retrieval': catalog.get('_retrieval', {'url': CATALOG_URL, 'retrieved_at': None,
                                                          'response_sha256': None})}
