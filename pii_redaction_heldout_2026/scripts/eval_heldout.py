"""
Held-out PII redaction generalization eval — runner.

Runs a target redactor against the 150-question held-out suite
(gen / doc / multilingual / code / negative) and grades each output
with gpt-4o, producing per-question scores and a per-bucket summary.

Usage:
    python eval_heldout.py --target privacy-filter
    python eval_heldout.py --target privacy-filter-regex
    python eval_heldout.py --target ollama:gemma4:e2b
    python eval_heldout.py --target ollama:qwen3.5:0.8b
    python eval_heldout.py --target hf-token-class:<hf_repo_id>

Add new targets by writing an adapter in `redactors.py`. The orchestration
in this file is target-agnostic.

Environment:
    OPENAI_API_KEY     used for the gpt-4o judge.
    OPENROUTER_API_KEY fallback; routes the judge to openai/gpt-4o through
                       OpenRouter (same model, identical scoring).

Output:
    results/held_out_suite_<target>.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(override=True)

QUESTIONS_DIR = ROOT / "evaluation_questions"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

HELD_OUT_BENCHES = {
    "gen": QUESTIONS_DIR / "generalization_bench.json",
    "doc": QUESTIONS_DIR / "doc_bench.json",
    "multilingual": QUESTIONS_DIR / "multilingual_bench.json",
    "code": QUESTIONS_DIR / "code_bench.json",
    "negative": QUESTIONS_DIR / "negative_bench.json",
}


# ─── Judge ────────────────────────────────────────────────────────────────

def get_grading_client():
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return OpenAI(api_key=api_key), "gpt-4o"
    or_key = os.environ.get("OPENROUTER_API_KEY")
    if not or_key:
        raise RuntimeError(
            "Need OPENAI_API_KEY or OPENROUTER_API_KEY in .env for the judge."
        )
    return OpenAI(api_key=or_key, base_url="https://openrouter.ai/api/v1"), "openai/gpt-4o"


def grade_redaction(client, judge, original, expected_pii, redacted):
    if expected_pii is None:
        expected_pii = []
    expected_str = (
        "\n".join(f"  - {p}" for p in expected_pii)
        if expected_pii
        else "  (none — this is a negative-control example; expected redaction is none)"
    )
    prompt = f"""
You are grading a privacy-pseudonymization model's output.

Original text:
{original}

Items that should have been redacted (ground truth):
{expected_str}

Model's redacted output:
{redacted}

Grade the model's redaction from 0 to 5:
- 5 = All ground-truth items were replaced by a placeholder, AND no non-PII content was redacted (no false positives).
- 4 = Almost all ground-truth items redacted; minor miss or one over-redaction.
- 3 = Most items redacted but multiple misses, OR one significant over-redaction.
- 2 = Half the items redacted, OR significant false-positive over-redaction.
- 1 = Few items redacted, OR many false positives.
- 0 = Failed completely (left obvious PII intact OR redacted purely non-PII text).

For negative-control questions where the expected list is empty, score:
- 5 = nothing redacted (correct)
- 0 = significant content redacted (false positives)

Be strict. Format your response as exactly:
Grade: <0-5>
Justification: <one sentence>
""".strip()
    r = client.chat.completions.create(
        model=judge,
        messages=[
            {"role": "system", "content": "You are a precise grader."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=512,
    )
    return r.choices[0].message.content.strip()


GRADE_RE = re.compile(r"Grade:\s*([0-9]+(?:\.[0-9]+)?)")
LEADING_NUM_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)")


def grade_to_pct(grade_text: str):
    m = GRADE_RE.search(grade_text) or LEADING_NUM_RE.search(grade_text)
    return (float(m.group(1)) / 5.0) * 100 if m else None


# ─── Bench runner ─────────────────────────────────────────────────────────

def run_one_bench(bench_name, bench_path, redactor, client, judge):
    bench = json.loads(bench_path.read_text())
    questions = bench["questions"]
    print(f"\n[bench:{bench_name}] {len(questions)} questions")

    rows = []
    for q in questions:
        text = q["text"]
        expected = q.get("expected_pii", [])
        t0 = time.time()
        try:
            redacted = redactor(text)
        except Exception as e:
            redacted = f"[ERROR: {type(e).__name__}: {str(e)[:120]}]"
            grade_text = f"Grade: 0\nJustification: {type(e).__name__}"
            rows.append({
                "id": q["id"],
                "context": q.get("context") or q.get("category") or q.get("locale", ""),
                "expected_count": len(expected),
                "score_pct": 0.0,
                "feedback": grade_text,
                "redacted_excerpt": redacted[:300],
                "infer_seconds": round(time.time() - t0, 2),
            })
            print(f"  Q{q['id']:>3}  ERROR  {type(e).__name__}")
            continue
        infer_s = time.time() - t0
        grade_text = grade_redaction(client, judge, text, expected, redacted)
        pct = grade_to_pct(grade_text)
        ctx = q.get("context") or q.get("doctype") or q.get("category") or q.get("locale", "")
        rows.append({
            "id": q["id"],
            "context": ctx,
            "expected_count": len(expected),
            "score_pct": pct,
            "feedback": grade_text,
            "redacted_excerpt": redacted[:300],
            "infer_seconds": round(infer_s, 2),
        })
        print(f"  Q{q['id']:>3} [{ctx[:35]:>35}] expected={len(expected):>2} pct={pct}")

    nums = [r["score_pct"] for r in rows if r["score_pct"] is not None]
    overall = round(sum(nums) / len(nums), 2) if nums else None
    print(f"  → {bench_name}: {overall}")
    return overall, rows


def load_target(target: str):
    """Resolve --target string to a callable redactor: (text:str) -> str."""
    from redactors import build_redactor
    return build_redactor(target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--target",
        required=True,
        help=(
            "Redactor target. Supported: 'privacy-filter', "
            "'privacy-filter-regex', 'ollama:<model>', 'hf-token-class:<repo>'. "
            "See redactors.py to add new targets."
        ),
    )
    ap.add_argument(
        "--label",
        default=None,
        help="Filename label (default: derive from --target).",
    )
    args = ap.parse_args()
    label = args.label or args.target.replace("/", "_").replace(":", "_")

    print(f"[heldout-eval] target={args.target}")
    redactor = load_target(args.target)

    client, judge = get_grading_client()
    print(f"[heldout-eval] grading via {judge}")

    suite = {}
    for bench_name, bench_path in HELD_OUT_BENCHES.items():
        t0 = time.time()
        overall, rows = run_one_bench(bench_name, bench_path, redactor, client, judge)
        suite[bench_name] = {"overall": overall, "rows": rows}
        print(f"  ({time.time() - t0:.1f}s)")

    print("\n" + "=" * 60)
    print(f"[heldout-eval] {label}")
    print("=" * 60)
    for bn, res in suite.items():
        print(f"  {bn:>14}: {res['overall']}")
    nums = [r["overall"] for r in suite.values() if r["overall"] is not None]
    if nums:
        print(f"  {'mean':>14}: {round(sum(nums) / len(nums), 2)}")

    out_path = RESULTS_DIR / f"held_out_suite_{label}.json"
    out_path.write_text(json.dumps({
        "label": label,
        "target": args.target,
        "benches": suite,
    }, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
