"""
HIPAA Limited Data Set (§164.514(e)(2)) compliance benchmark — runner.

Wooden, deterministic scoring. Unlike PrivacyBench's gpt-4o subjective
grader, this benchmark is rule-based: every email is PHI, every phone
is PHI, every name is PHI. The benchmark also tests "retained" carve-outs
(city/state/ZIP, dates, ages) that under HIPAA LDS MUST be preserved.

Per-question score (0-5):
    5  perfect: all PHI removed with right LDS class, all retained signals preserved
    4  all PHI removed but 1 class miscategorized OR 1 retained-signal over-redaction
    3  all PHI removed but ≥2 issues
    2  ≥1 PHI missed (HIPAA violation)
    1  ≥2 PHI missed
    0  catastrophic (everything missed or text destroyed)

Top-line metrics:
    overall_score_pct       arithmetic mean of (score/5*100)
    hipaa_recall_pct        % of expected PHI spans that were caught (any class)
    carveout_preservation_pct  % of retained_signals that the model did NOT redact

For HIPAA defensibility, hipaa_recall_pct must be 100%.

Usage:
    python eval_hipaa_lds.py --target privacy-filter
    python eval_hipaa_lds.py --target hf-token-class:<repo>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

EVAL_FILE = ROOT / "evaluation_questions" / "hipaa_lds_bench.json"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def overlap(a_start, a_end, b_start, b_end) -> bool:
    return not (a_end <= b_start or a_start >= b_end)


def score_question(q, predicted_spans):
    """Apply the rubric in evaluation_questions/hipaa_lds_bench.json."""
    expected_phi = q.get("phi_spans", [])
    expected_retained = q.get("retained_signals", [])

    caught_phi = []
    missed_phi = []
    matched = set()
    for exp in expected_phi:
        match_idx = None
        for j, pred in enumerate(predicted_spans):
            if j in matched:
                continue
            if overlap(exp["start"], exp["end"], pred["start"], pred["end"]):
                match_idx = j
                break
        if match_idx is None:
            missed_phi.append(exp)
        else:
            caught_phi.append({"expected": exp, "predicted": predicted_spans[match_idx]})
            matched.add(match_idx)

    class_correct = sum(1 for c in caught_phi if c["expected"]["label"] == c["predicted"]["label"])
    class_wrong = len(caught_phi) - class_correct

    retained_violations = []
    for ret in expected_retained:
        for j, pred in enumerate(predicted_spans):
            if overlap(ret["start"], ret["end"], pred["start"], pred["end"]):
                retained_violations.append({"retained": ret, "redacted_as": pred})
                break

    spurious = []
    for j, pred in enumerate(predicted_spans):
        if j in matched:
            continue
        if any(overlap(ret["start"], ret["end"], pred["start"], pred["end"]) for ret in expected_retained):
            continue
        spurious.append(pred)

    n_missed = len(missed_phi)
    n_violations = len(retained_violations)
    n_spurious = len(spurious)
    if n_missed >= 2:
        score = 1
    elif n_missed >= 1:
        score = 2
    elif n_violations >= 2 or class_wrong >= 2:
        score = 3
    elif n_violations == 1 or class_wrong == 1:
        score = 4
    else:
        score = 5
    if n_spurious >= 3 and score == 5:
        score = 4

    return {
        "id": q["id"],
        "category": q["category"],
        "score": score,
        "n_expected_phi": len(expected_phi),
        "n_caught": len(caught_phi),
        "n_missed": n_missed,
        "n_class_wrong": class_wrong,
        "n_retained_violations": n_violations,
        "n_spurious": n_spurious,
        "missed_phi": missed_phi,
        "retained_violations": retained_violations,
        "spurious_redactions": spurious,
    }


def aggregate(per_q):
    total_phi = sum(q["n_expected_phi"] for q in per_q)
    total_caught = sum(q["n_caught"] for q in per_q)
    total_retained = sum(q["n_retained_violations"] for q in per_q)
    total_retained_expected = sum(
        q["n_retained_violations"] + (q["n_expected_phi"] - q["n_missed"]) - 0
        for q in per_q
    )
    # Simpler: count retained-signal preservation directly from raw eval
    overall = round(sum(q["score"] for q in per_q) / len(per_q) / 5 * 100, 2)
    hipaa_recall = round(100 * total_caught / total_phi, 2) if total_phi else None
    return {"overall_score_pct": overall, "hipaa_recall_pct": hipaa_recall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--target",
        required=True,
        help="Detector target. See ../../pii_redaction_heldout_2026/scripts/redactors.py for adapter conventions.",
    )
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    label = args.label or args.target.replace("/", "_").replace(":", "_")

    # Re-use the redactor adapters from the heldout-2026 task — same shape:
    # build_span_predictor(target) -> Callable[[str], list[{start,end,label}]]
    sys.path.insert(0, str(ROOT.parent / "pii_redaction_heldout_2026" / "scripts"))
    try:
        from redactors_spans import build_span_predictor
    except ImportError:
        print(
            "ERROR: this benchmark requires a span-level predictor (start/end/label),\n"
            "not a redacted-text output. See `redactors_spans.py` for adapter conventions."
        )
        sys.exit(2)

    predictor = build_span_predictor(args.target)

    eval_data = json.loads(EVAL_FILE.read_text())
    questions = eval_data["questions"]
    print(f"[hipaa] {len(questions)} questions, target={args.target}")

    per_q = []
    for q in questions:
        spans = predictor(q["text"])
        per_q.append(score_question(q, spans))

    summary = aggregate(per_q)

    # Carve-out preservation is computed end-to-end against the eval data.
    total_retained = sum(len(q.get("retained_signals", [])) for q in questions)
    total_violations = sum(p["n_retained_violations"] for p in per_q)
    summary["carveout_preservation_pct"] = (
        round(100 * (total_retained - total_violations) / total_retained, 2)
        if total_retained else None
    )

    print(f"\n  overall_score_pct:        {summary['overall_score_pct']}")
    print(f"  hipaa_recall_pct:         {summary['hipaa_recall_pct']}")
    print(f"  carveout_preservation_pct: {summary['carveout_preservation_pct']}")

    out = RESULTS_DIR / f"hipaa_lds_eval_{label}.json"
    out.write_text(json.dumps({
        "label": label,
        "target": args.target,
        "summary": summary,
        "per_question": per_q,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
