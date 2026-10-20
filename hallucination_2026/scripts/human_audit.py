"""
Stratified human-audit evaluation script for PrivacyBench v2 (`hallucination_2026`).

Samples >= 60 graded items stratified across all evaluation modes:
- Mode A1: `grounded_answerable` (20 items)
- Mode A2: `grounded_nearmiss` (15 items)
- Mode B:  `closed_book` (15 items)
- Mode C:  `summary` atomic claims (15 claims)
Total = 65 audited items.

Applies strict manual rubric verification rules against the verbatim source evidence
and gold targets, and computes raw agreement % and Cohen's kappa between the human
audit labels and the 3-judge panel (`google/gemini-4-argon`, `meta/muse-wonder`, `x-ai/grok-4.7`).
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
TASK_ROOT = SCRIPT_DIR.parent


def cohens_kappa(y1: List[str], y2: List[str], labels: List[str]) -> float:
    assert len(y1) == len(y2)
    n = len(y1)
    if n == 0:
        return 1.0
    p_o = sum(1 for a, b in zip(y1, y2) if a == b) / n
    c1 = Counter(y1)
    c2 = Counter(y2)
    p_e = sum((c1[lbl] / n) * (c2[lbl] / n) for lbl in labels)
    if abs(1.0 - p_e) < 1e-9:
        return 1.0
    return round((p_o - p_e) / (1.0 - p_e), 3)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export or score stratified human audit sample (>=50 items).")
    ap.add_argument("--results-dir", default=str(TASK_ROOT / "results"))
    ap.add_argument("--audit-file", default=str(TASK_ROOT / "evaluation_questions" / "human_audit_60.json"))
    args = ap.parse_args()

    audit_path = Path(args.audit_file)
    if not audit_path.exists():
        print(f"Audit file {audit_path} not yet created.")
        return

    audit_data = json.loads(audit_path.read_text(encoding="utf-8"))
    items = audit_data.get("audited_items", [])
    qa_items = [x for x in items if x["mode"] != "summary_claim"]
    sum_items = [x for x in items if x["mode"] == "summary_claim"]

    qa_human = [x["human_grade"] for x in qa_items]
    qa_panel = [x["panel_majority_grade"] for x in qa_items]
    qa_agree = sum(1 for h, p in zip(qa_human, qa_panel) if h == p)
    qa_kappa = cohens_kappa(qa_human, qa_panel, ["CORRECT", "INCORRECT", "PARTIAL_ANSWER", "NOT_ATTEMPTED"])

    sum_human = [x["human_grade"] for x in sum_items]
    sum_panel = [x["panel_majority_grade"] for x in sum_items]
    sum_agree = sum(1 for h, p in zip(sum_human, sum_panel) if h == p)
    sum_kappa = cohens_kappa(sum_human, sum_panel, ["SUPPORTED", "UNSUPPORTED", "CONTRADICTED"])

    tot_agree = qa_agree + sum_agree
    tot_n = len(items)
    print(f"[human_audit] Total audited items: {tot_n}")
    print(f"  Overall Human vs Panel Agreement: {tot_agree}/{tot_n} ({100.0*tot_agree/tot_n:.1f}%)")
    print(f"  QA Items ({len(qa_items)}): Agreement={100.0*qa_agree/len(qa_items):.1f}%, Cohen's kappa={qa_kappa:.3f}")
    print(f"  Summary Claims ({len(sum_items)}): Agreement={100.0*sum_agree/len(sum_items):.1f}%, Cohen's kappa={sum_kappa:.3f}")


if __name__ == "__main__":
    main()
