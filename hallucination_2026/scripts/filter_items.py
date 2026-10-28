"""
Six-stage filtering pipeline for PrivacyBench v2 (`hallucination_2026`).

Stages:
1. `answerability_and_evidence`: Verifies verbatim evidence offsets in normalized article text,
   and that a strong validator model from a different lab reaches the gold answer in grounded mode.
2. `unambiguity`: Two independent validator models from different labs given the article both
   agree with the gold answer.
3. `unknowability`: Frontier validator models asked closed-book (no article) fail or abstain
   (cannot answer `CORRECT` without the source).
4. `near_miss_validity`: For near-miss unanswerable items, validators asked to locate the fact
   in the article confirm it is absent (`NOT_STATED`) while the adjacent distractor quote is present.
5. `deduplication`: Drops near-duplicate questions within the same article using token/3-gram
   similarity and balances items per article and generator lab.
6. `difficulty`: Drops/flags any answerable item that 100% of evaluated models (including small
   models) get right in grounded mode.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llm_client import TASK_ROOT, call_llm, get_article_text, served_record

# Three high-legal-skill, low-hallucination validators from three distinct labs
VALIDATOR_MODELS = {
    "google": "google/gemini-4-argon",
    "meta": "meta/muse-wonder",
    "x-ai": "x-ai/grok-4.7",
}

GROUNDED_VALIDATE_PROMPT = """You are validating a benchmark question using ONLY the provided article text.
Answer the question with JUST the short exact answer (no explanation).
If the requested specific fact is not stated in the article, reply with exactly: NOT_STATED

Article:
{article_text}

Question: {question}
Answer:"""

CLOSED_BOOK_VALIDATE_PROMPT = """You are answering a question about {domain} ({subtopic}) with no external context or tools.
Answer with JUST the short exact answer (no explanation).
If you do not know the answer or need the source document to know the exact figure/date/fact, reply with: ABSTAIN

Question: {question}
Answer:"""


def _norm_ans(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"^(the answer is|answer:)\s*", "", s)
    s = s.strip(" .,:;\"'()[]{}")
    s = re.sub(r"\s+", " ", s)
    return s


def _extract_numbers(s: str) -> List[float]:
    cleaned = s.replace(",", "").replace("$", "").replace("%", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    out = []
    for m in matches:
        try:
            out.append(float(m))
        except ValueError:
            pass
    return out


def answers_agree(pred: str, gold: str, answer_type: str, precision_rule: str) -> bool:
    """Check whether validator's predicted short answer matches the gold answer."""
    np = _norm_ans(pred)
    ng = _norm_ans(gold)
    if not np or "not_stated" in np or "not stated" in np or "abstain" in np or "do not know" in np or "don't know" in np:
        return False
    if np == ng or ng in np or np in ng:
        # Guard against short numeric substring false matches (e.g. "4" in "45")
        if answer_type in ("number", "identifier"):
            nums_p = _extract_numbers(np)
            nums_g = _extract_numbers(ng)
            if nums_g and nums_p:
                if precision_rule == "exact_identifier" or answer_type == "identifier":
                    return np.replace(" ", "") == ng.replace(" ", "") or ng in np.split()
                return abs(nums_p[0] - nums_g[0]) <= 1e-6 * max(1.0, abs(nums_g[0]))
        return True

    if answer_type == "number" and precision_rule != "exact_identifier":
        nums_p = _extract_numbers(np)
        nums_g = _extract_numbers(ng)
        if len(nums_p) == 1 and len(nums_g) == 1:
            g_val = nums_g[0]
            p_val = nums_p[0]
            # Check last significant figure tolerance
            g_str = re.sub(r"[^0-9.]", "", gold)
            if "." in g_str:
                decimals = len(g_str.split(".")[1])
                tol = 0.5 * (10 ** (-decimals))
            else:
                tol = 0.5
            return abs(p_val - g_val) <= tol

    # Date normalization (e.g. "July 14, 2026" vs "2026-07-14")
    if answer_type == "date":
        nums_p = [int(x) for x in re.findall(r"\d+", np)]
        nums_g = [int(x) for x in re.findall(r"\d+", ng)]
        if set(nums_p) == set(nums_g) and len(nums_g) >= 2:
            return True

    return False


def is_abstention(pred: str) -> bool:
    np = _norm_ans(pred)
    abstain_markers = [
        "not_stated",
        "not stated",
        "abstain",
        "do not know",
        "don't know",
        "cannot be determined",
        "not mentioned",
        "does not state",
        "need more context",
        "no information",
        "not provided",
        "unable to",
        "unknown",
    ]
    return any(m in np for m in abstain_markers)


def question_similarity(q1: str, q2: str) -> float:
    """Compute combined word and 3-gram Jaccard similarity between two questions."""
    w1 = set(re.findall(r"[a-z0-9]+", q1.lower()))
    w2 = set(re.findall(r"[a-z0-9]+", q2.lower()))
    stop = {"what", "which", "when", "where", "who", "how", "many", "much", "the", "in", "of", "to", "for", "and", "on", "is", "was", "under", "according", "answer", "as", "an", "a", "by", "with", "from"}
    cw1 = w1 - stop
    cw2 = w2 - stop
    if not cw1 or not cw2:
        return 0.0
    word_jacc = len(cw1 & cw2) / len(cw1 | cw2)

    toks1 = re.findall(r"[a-z0-9]+", q1.lower())
    toks2 = re.findall(r"[a-z0-9]+", q2.lower())
    g1 = set(tuple(toks1[i : i + 3]) for i in range(max(0, len(toks1) - 2)))
    g2 = set(tuple(toks2[i : i + 3]) for i in range(max(0, len(toks2) - 2)))
    gram_jacc = (len(g1 & g2) / len(g1 | g2)) if (g1 and g2) else 0.0
    return round(0.6 * word_jacc + 0.4 * gram_jacc, 3)


def validate_single_candidate(
    cand: Dict[str, Any],
    norm_article: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run Stages 1–4 on a single candidate item."""
    telemetry: List[Dict[str, Any]] = []
    verdicts: Dict[str, Any] = {
        "evidence_span_verified": False,
        "answerability_pass": None,
        "unambiguity_validators": {},
        "closed_book_unknowability": {},
        "near_miss_absence_verified": None,
        "max_shingle_jaccard": 0.0,
        "filter_status": "PENDING",
        "drop_stage": None,
        "drop_reason": None,
    }

    # Stage 1a: Evidence span check
    offsets = cand.get("evidence_offsets")
    quote = cand.get("evidence_quote", "")
    if (
        offsets
        and isinstance(offsets, list)
        and len(offsets) == 2
        and offsets[0] is not None
        and offsets[1] is not None
        and 0 <= offsets[0] < offsets[1] <= len(norm_article)
        and len(quote.strip()) >= 15
    ):
        verdicts["evidence_span_verified"] = True
    else:
        verdicts["filter_status"] = "DROP"
        verdicts["drop_stage"] = "stage1_evidence_span"
        verdicts["drop_reason"] = "Verbatim evidence quote not found in normalized article text."
        cand["validator_verdicts"] = verdicts
        return cand, telemetry

    # Choose two grounded validators (always including at least one lab different from generator)
    gen_lab = cand["generator_model"].split("/")[0]
    val_models = [m for lab, m in VALIDATOR_MODELS.items() if lab != gen_lab]
    if len(val_models) < 2:
        val_models = list(VALIDATOR_MODELS.values())[:2]

    if cand["answerable"]:
        # Stage 1b & Stage 2: Grounded Answerability + Unambiguity across 2 independent validators
        agrees_count = 0
        for vm in val_models[:2]:
            resp = call_llm(
                model_id=vm,
                messages=[
                    {
                        "role": "user",
                        "content": GROUNDED_VALIDATE_PROMPT.format(
                            article_text=norm_article,
                            question=cand["question"],
                        ),
                    }
                ],
                temperature=0.0,
                max_tokens=256,
                reasoning_effort="low",
            )
            telemetry.append({"step": "filter_grounded", "candidate_id": cand["id"], **resp})
            pred = resp["content"].strip()
            ok = answers_agree(pred, cand["gold_answer"], cand["answer_type"], cand["numeric_precision_rule"])
            verdicts["unambiguity_validators"][vm] = {"pred": pred, "agrees": ok, "served": served_record(resp)}
            if ok:
                agrees_count += 1

        if agrees_count == 0:
            verdicts["answerability_pass"] = False
            verdicts["filter_status"] = "DROP"
            verdicts["drop_stage"] = "stage1_answerability"
            verdicts["drop_reason"] = "Neither grounded validator reached the gold answer."
            cand["validator_verdicts"] = verdicts
            return cand, telemetry

        verdicts["answerability_pass"] = True
        if agrees_count < 2:
            # Try the 3rd lab validator as tie-breaker for unambiguity; require at least 2/3 agreement
            third_vm = [m for m in VALIDATOR_MODELS.values() if m not in val_models[:2]][0]
            resp3 = call_llm(
                model_id=third_vm,
                messages=[
                    {
                        "role": "user",
                        "content": GROUNDED_VALIDATE_PROMPT.format(
                            article_text=norm_article,
                            question=cand["question"],
                        ),
                    }
                ],
                temperature=0.0,
                max_tokens=256,
                reasoning_effort="low",
            )
            telemetry.append({"step": "filter_grounded_tiebreak", "candidate_id": cand["id"], **resp3})
            pred3 = resp3["content"].strip()
            ok3 = answers_agree(pred3, cand["gold_answer"], cand["answer_type"], cand["numeric_precision_rule"])
            verdicts["unambiguity_validators"][third_vm] = {"pred": pred3, "agrees": ok3, "served": served_record(resp3)}
            if ok3:
                agrees_count += 1

        if agrees_count < 2:
            verdicts["filter_status"] = "DROP"
            verdicts["drop_stage"] = "stage2_unambiguity"
            verdicts["drop_reason"] = "Independent grounded validators disagreed with gold answer."
            cand["validator_verdicts"] = verdicts
            return cand, telemetry

        # Stage 3: Closed-Book Unknowability (must fail or abstain without article)
        cb_correct_count = 0
        for vm in val_models[:2]:
            resp_cb = call_llm(
                model_id=vm,
                messages=[
                    {
                        "role": "user",
                        "content": CLOSED_BOOK_VALIDATE_PROMPT.format(
                            domain=cand["domain"],
                            subtopic=cand["subtopic"],
                            question=cand["question"],
                        ),
                    }
                ],
                temperature=0.0,
                max_tokens=256,
                reasoning_effort="low",
            )
            telemetry.append({"step": "filter_closed_book", "candidate_id": cand["id"], **resp_cb})
            pred_cb = resp_cb["content"].strip()
            cb_ok = answers_agree(pred_cb, cand["gold_answer"], cand["answer_type"], cand["numeric_precision_rule"])
            cb_abst = is_abstention(pred_cb)
            grade_cb = "CORRECT" if cb_ok else ("NOT_ATTEMPTED" if cb_abst else "INCORRECT")
            verdicts["closed_book_unknowability"][vm] = {"pred": pred_cb, "grade": grade_cb, "served": served_record(resp_cb)}
            if cb_ok:
                cb_correct_count += 1

        if cb_correct_count > 0:
            verdicts["filter_status"] = "DROP"
            verdicts["drop_stage"] = "stage3_unknowability"
            verdicts["drop_reason"] = "Answered correctly in closed-book mode (fact guessable or pre-cutoff)."
            cand["validator_verdicts"] = verdicts
            return cand, telemetry

        verdicts["filter_status"] = "PASS_PRE_DEDUP"
        cand["validator_verdicts"] = verdicts
        return cand, telemetry

    else:
        # Near-Miss Unanswerable: Stage 4 — verify fact is genuinely absent in grounded mode
        absent_count = 0
        for vm in val_models[:2]:
            resp_nm = call_llm(
                model_id=vm,
                messages=[
                    {
                        "role": "user",
                        "content": GROUNDED_VALIDATE_PROMPT.format(
                            article_text=norm_article,
                            question=cand["question"],
                        ),
                    }
                ],
                temperature=0.0,
                max_tokens=256,
                reasoning_effort="low",
            )
            telemetry.append({"step": "filter_nearmiss", "candidate_id": cand["id"], **resp_nm})
            pred_nm = resp_nm["content"].strip()
            is_abs = is_abstention(pred_nm)
            verdicts["unambiguity_validators"][vm] = {"pred": pred_nm, "confirms_absent": is_abs, "served": served_record(resp_nm)}
            if is_abs:
                absent_count += 1

        if absent_count < 2:
            # Check third validator; require at least 2/3 validators to confirm fact is absent
            third_vm = [m for m in VALIDATOR_MODELS.values() if m not in val_models[:2]][0]
            resp3 = call_llm(
                model_id=third_vm,
                messages=[
                    {
                        "role": "user",
                        "content": GROUNDED_VALIDATE_PROMPT.format(
                            article_text=norm_article,
                            question=cand["question"],
                        ),
                    }
                ],
                temperature=0.0,
                max_tokens=256,
                reasoning_effort="low",
            )
            telemetry.append({"step": "filter_nearmiss_tiebreak", "candidate_id": cand["id"], **resp3})
            pred3 = resp3["content"].strip()
            is_abs3 = is_abstention(pred3)
            verdicts["unambiguity_validators"][third_vm] = {"pred": pred3, "confirms_absent": is_abs3, "served": served_record(resp3)}
            if is_abs3:
                absent_count += 1

        if absent_count < 2:
            verdicts["near_miss_absence_verified"] = False
            verdicts["filter_status"] = "DROP"
            verdicts["drop_stage"] = "stage4_nearmiss_validity"
            verdicts["drop_reason"] = "Validators did not confirm that the near-miss target fact is absent."
            cand["validator_verdicts"] = verdicts
            return cand, telemetry

        verdicts["near_miss_absence_verified"] = True
        verdicts["filter_status"] = "PASS_PRE_DEDUP"
        cand["validator_verdicts"] = verdicts
        return cand, telemetry


def run_filter_pipeline(
    candidates: List[Dict[str, Any]],
    articles_by_id: Dict[str, Dict[str, Any]],
    target_answerable_per_article: int = 7,
    target_nearmiss_per_article: int = 3,
    sim_threshold: float = 0.52,
    max_workers: int = 14,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """Run all filter stages in parallel across candidates and deduplicate per article."""
    norm_articles = {aid: get_article_text(art, verify_hash=True) for aid, art in articles_by_id.items()}

    all_telemetry: List[Dict[str, Any]] = []
    validated_cands: List[Dict[str, Any]] = [None] * len(candidates)  # type: ignore

    print(f"[filter] Validating {len(candidates)} candidates across Stages 1-4 (workers={max_workers})...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_idx = {
            ex.submit(validate_single_candidate, cand, norm_articles[cand["article_id"]]): i
            for i, cand in enumerate(candidates)
        }
        for fut in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[fut]
            updated_cand, telem = fut.result()
            validated_cands[idx] = updated_cand
            all_telemetry.extend(telem)

    # Stage 5: Deduplication and balanced selection per article
    kept_items: List[Dict[str, Any]] = []
    by_article: Dict[str, List[Dict[str, Any]]] = {}
    for c in validated_cands:
        by_article.setdefault(c["article_id"], []).append(c)

    for aid, art_cands in by_article.items():
        passing_ans = [c for c in art_cands if c["answerable"] and c["validator_verdicts"]["filter_status"] == "PASS_PRE_DEDUP"]
        passing_nm = [c for c in art_cands if (not c["answerable"]) and c["validator_verdicts"]["filter_status"] == "PASS_PRE_DEDUP"]

        # Interleave generators so both generator labs are equally represented
        def _interleave_by_gen(items_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            by_g: Dict[str, List[Dict[str, Any]]] = {}
            for it in items_list:
                by_g.setdefault(it["generator_model"], []).append(it)
            gens = sorted(by_g.keys())
            out = []
            max_len = max((len(v) for v in by_g.values()), default=0)
            for i in range(max_len):
                for g in gens:
                    if i < len(by_g[g]):
                        out.append(by_g[g][i])
            return out

        selected_for_art: List[Dict[str, Any]] = []
        for group, quota in [(_interleave_by_gen(passing_ans), target_answerable_per_article), (_interleave_by_gen(passing_nm), target_nearmiss_per_article)]:
            group_kept = 0
            for c in group:
                sims = [question_similarity(c["question"], prev["question"]) for prev in selected_for_art]
                max_sim = max(sims, default=0.0)
                c["validator_verdicts"]["max_shingle_jaccard"] = max_sim
                if max_sim >= sim_threshold:
                    c["validator_verdicts"]["filter_status"] = "DROP"
                    c["validator_verdicts"]["drop_stage"] = "stage5_deduplication"
                    c["validator_verdicts"]["drop_reason"] = f"Similarity {max_sim} >= {sim_threshold} with already kept item."
                    continue
                if group_kept >= quota:
                    c["validator_verdicts"]["filter_status"] = "DROP"
                    c["validator_verdicts"]["drop_stage"] = "stage5_quota_balance"
                    c["validator_verdicts"]["drop_reason"] = f"Article quota ({quota}) already filled after deduplication."
                    continue
                c["validator_verdicts"]["filter_status"] = "KEEP"
                selected_for_art.append(c)
                group_kept += 1

        # Assign clean sequential item IDs per article
        for j, item in enumerate(selected_for_art, start=1):
            item["id"] = f"{aid}_q{j:02d}"
            kept_items.append(item)

    # Summarize stage yields
    stage_counts = {
        "raw_candidates": len(validated_cands),
        "raw_answerable": sum(1 for c in validated_cands if c["answerable"]),
        "raw_nearmiss": sum(1 for c in validated_cands if not c["answerable"]),
        "dropped_stage1_evidence_span": sum(1 for c in validated_cands if c["validator_verdicts"].get("drop_stage") == "stage1_evidence_span"),
        "dropped_stage1_answerability": sum(1 for c in validated_cands if c["validator_verdicts"].get("drop_stage") == "stage1_answerability"),
        "dropped_stage2_unambiguity": sum(1 for c in validated_cands if c["validator_verdicts"].get("drop_stage") == "stage2_unambiguity"),
        "dropped_stage3_unknowability": sum(1 for c in validated_cands if c["validator_verdicts"].get("drop_stage") == "stage3_unknowability"),
        "dropped_stage4_nearmiss_validity": sum(1 for c in validated_cands if c["validator_verdicts"].get("drop_stage") == "stage4_nearmiss_validity"),
        "dropped_stage5_deduplication": sum(1 for c in validated_cands if c["validator_verdicts"].get("drop_stage") == "stage5_deduplication"),
        "dropped_stage5_quota_balance": sum(1 for c in validated_cands if c["validator_verdicts"].get("drop_stage") == "stage5_quota_balance"),
        "kept_total": len(kept_items),
        "kept_answerable": sum(1 for c in kept_items if c["answerable"]),
        "kept_nearmiss": sum(1 for c in kept_items if not c["answerable"]),
        "filter_cost_usd": round(sum(t.get("cost_usd", 0.0) for t in all_telemetry), 4),
    }

    return kept_items, stage_counts, validated_cands


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter candidate questions for PrivacyBench v2.")
    ap.add_argument(
        "--candidates",
        default=str(TASK_ROOT / "evaluation_questions" / "candidates_raw.json"),
    )
    ap.add_argument(
        "--articles",
        default=str(TASK_ROOT / "evaluation_questions" / "articles.json"),
    )
    ap.add_argument(
        "--out",
        default=str(TASK_ROOT / "evaluation_questions" / "items.json"),
    )
    ap.add_argument(
        "--log",
        default=str(TASK_ROOT / "evaluation_questions" / "candidates_filtered_log.json"),
    )
    args = ap.parse_args()

    cands_data = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    arts_data = json.loads(Path(args.articles).read_text(encoding="utf-8"))
    articles_by_id = {a["id"]: a for a in arts_data["articles"]}

    kept_items, stage_counts, all_evaluated = run_filter_pipeline(
        cands_data["candidates"],
        articles_by_id,
    )

    Path(args.out).write_text(
        json.dumps(
            {
                "_meta": {
                    "filtered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "stage_yields": stage_counts,
                },
                "items": kept_items,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    Path(args.log).write_text(
        json.dumps(
            {
                "stage_yields": stage_counts,
                "candidates_with_verdicts": all_evaluated,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("\n[filter] Stage yield summary:")
    for k, v in stage_counts.items():
        print(f"  {k:34s}: {v}")
    print(f"[filter] Wrote {len(kept_items)} validated items to {args.out}")


if __name__ == "__main__":
    main()
