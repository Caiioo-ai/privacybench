"""Per-call prompt-token audit of a PrivacyBench result: finds calls a gateway channel padded with its own prompt.

A model reads only the benchmark's prompt, so its billed prompt tokens are fixed by the item. Two checks:
- against a reference run of the same model (same tokenizer, e.g. its earlier OpenRouter run): within each prompt
  mode every item moves by one constant (a host's chat template: glm-5.3-flash's closed-book prompts sit 15 below its
  OpenRouter run) give or take host jitter (up to 32 on glm-5.3); a call further off than that was padded;
- self-consistency, where no reference exists: an article's grounded prompts differ only by their questions,
  closed-book prompts only by theirs, and each summary sits a constant below its article's grounded prompts.
  A channel can add to our prompt, never remove from it, so the baseline is the LOWEST call, not the median: when most
  of an article's calls were padded (qwen3.7-max on CometAPI, 2026-10-29: +320 on 4 of 7), a median re-queues the clean
  ones and keeps the padded. Clean runs spread up to 96 / 103 / 38 tokens from the lowest (max minus min, all 37
  results at 9f5ef22), hence the thresholds. A pad on EVERY call of an article is invisible here; that needs a
  reference run.

Measured 2026-10-28 on CometAPI: a Codex-backed channel adds ~4,216 tokens (it also echoes them and is refused at
call time) and a gemini-4-argon channel adds ~132 without saying so. `--requeue` turns flagged rows into error rows,
which `run_eval.py` re-runs on its next pass over the same output.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

REFERENCE_THRESHOLD = 60
# A per-mode constant up to this is a host's chat template (minimax-m3.1-flash +19); beyond it, a hidden prompt on
# every call (mimo-v2.6-flash on CometAPI: +792), so the whole mode is flagged.
TEMPLATE_TOLERANCE = 20
SELF_THRESHOLDS = {'grounded': 120, 'closed_book': 130, 'summary': 60}


def _rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return data.get('qa_responses', []) + data.get('summary_responses', [])


def _ok(row: Dict[str, Any]) -> bool:
    return not str(row.get('response', '')).startswith('[ERROR')


def _key(row: Dict[str, Any]) -> tuple:
    return (row['eval_mode'], row['item_id'])


def _tokens(row: Dict[str, Any]) -> int:
    return int(row['telemetry']['prompt_tokens'])


def _family(eval_mode: str) -> str:
    return 'grounded' if eval_mode.startswith('grounded') else eval_mode


def against_reference(new: Dict[str, Any], reference: Dict[str, Any], threshold: int = REFERENCE_THRESHOLD) -> List[dict]:
    ref = {_key(r): _tokens(r) for r in _rows(reference) if _ok(r)}
    deltas = {_key(r): _tokens(r) - ref[_key(r)] for r in _rows(new) if _ok(r) and _key(r) in ref}
    by_family = defaultdict(list)
    for key, delta in deltas.items():
        by_family[_family(key[0])].append(delta)
    offsets = {family: statistics.median(values) for family, values in by_family.items()}
    template = {family: offset if abs(offset) <= TEMPLATE_TOLERANCE else 0 for family, offset in offsets.items()}
    return [{'key': k, 'excess': d - template[_family(k[0])]} for k, d in sorted(deltas.items())
            if abs(d - template[_family(k[0])]) > threshold]


def self_consistency(data: Dict[str, Any], thresholds: Dict[str, int] = SELF_THRESHOLDS) -> List[dict]:
    rows = [r for r in _rows(data) if _ok(r)]
    grounded = defaultdict(list)
    for r in rows:
        if r['eval_mode'].startswith('grounded'):
            grounded[r['article_id']].append(_tokens(r))
    grounded_floor = {a: min(v) for a, v in grounded.items()}
    flagged = []
    for r in rows:
        if r['eval_mode'].startswith('grounded'):
            excess = _tokens(r) - grounded_floor[r['article_id']]
            if excess > thresholds['grounded']:
                flagged.append({'key': _key(r), 'excess': excess})
    closed = [r for r in rows if r['eval_mode'] == 'closed_book']
    if closed:
        floor = min(_tokens(r) for r in closed)
        flagged += [{'key': _key(r), 'excess': _tokens(r) - floor} for r in closed if _tokens(r) - floor > thresholds['closed_book']]
    summaries = [r for r in rows if r['eval_mode'] == 'summary' and r['article_id'] in grounded_floor]
    if summaries:
        offsets = {_key(r): _tokens(r) - grounded_floor[r['article_id']] for r in summaries}
        floor = min(offsets.values())
        flagged += [{'key': k, 'excess': o - floor} for k, o in offsets.items() if o - floor > thresholds['summary']]
    return sorted(flagged, key=lambda f: f['key'])


def requeue(data: Dict[str, Any], flagged: List[dict]) -> int:
    """Turn each flagged row into an error row (no grade, no cost), so the next run_eval pass sends it again."""
    keys = {f['key']: f['excess'] for f in flagged}
    changed = 0
    for row in _rows(data):
        if _key(row) in keys:
            telemetry = row.get('telemetry', {})
            row['response'] = f"[ERROR: PromptAudit: prompt {telemetry.get('prompt_tokens')} tokens, {keys[_key(row)]:+} off this model's own]"
            row['telemetry'] = {**telemetry, 'cost_usd': 0.0, 'provider': 'error', 'error': 'prompt-token audit'}
            row.pop('grading', None)
            changed += 1
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('result', help='Result JSON to audit')
    ap.add_argument('--reference', default=None, help="An earlier run of the same model (exact per-item check)")
    ap.add_argument('--requeue', action='store_true', help='Rewrite flagged rows as error rows for the next pass')
    args = ap.parse_args()
    path = Path(args.result)
    data = json.loads(path.read_text(encoding='utf-8'))
    flagged = (against_reference(data, json.loads(Path(args.reference).read_text(encoding='utf-8')))
               if args.reference else self_consistency(data))
    print(json.dumps({'model_id': data['model_meta']['model_id'], 'mode': 'reference' if args.reference else 'self',
                      'flagged': [{'eval_mode': f['key'][0], 'item_id': f['key'][1], 'excess': f['excess']} for f in flagged]}))
    if args.requeue and flagged:
        requeue(data, flagged)
        if 'grading_summary' in data:
            del data['grading_summary']
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
