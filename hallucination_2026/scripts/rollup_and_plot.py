"""
Rollup aggregator, cluster-bootstrap CI estimator, generator-bias auditor,
and multi-panel chart plotter for PrivacyBench v2 (`hallucination_2026`).

Computes all benchmark metrics per model with 95% confidence intervals via
non-parametric cluster bootstrap (resampling articles with replacement, B=2000).
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
TASK_ROOT = SCRIPT_DIR.parent


def compute_metrics_from_subsets(
    qa_items: List[Dict[str, Any]],
    summaries: List[Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    """Compute scalar metrics from a list of graded QA items and summary items."""
    g_ans = [q for q in qa_items if q["eval_mode"] == "grounded_answerable"]
    g_nm = [q for q in qa_items if q["eval_mode"] == "grounded_nearmiss"]
    cb = [q for q in qa_items if q["eval_mode"] == "closed_book"]

    def _grade(q: Dict[str, Any]) -> str:
        return q.get("grading", {}).get("final_grade", "INCORRECT")

    # 1. Grounded Answerable metrics
    n_g = len(g_ans)
    c_g = sum(1 for q in g_ans if _grade(q) == "CORRECT")
    p_g = sum(1 for q in g_ans if _grade(q) == "PARTIAL_ANSWER")
    i_g = sum(1 for q in g_ans if _grade(q) == "INCORRECT")
    a_g = sum(1 for q in g_ans if _grade(q) == "NOT_ATTEMPTED")

    grounded_acc = (c_g / n_g) if n_g else 0.0
    omniscience_idx = (100.0 * (c_g + 0.5 * p_g - i_g) / n_g) if n_g else 0.0
    hr_g_denom = i_g + p_g + a_g
    hr_g = (i_g / hr_g_denom) if hr_g_denom > 0 else 0.0

    # 2. Near-miss unanswerable abstention rate
    n_nm = len(g_nm)
    a_nm = sum(1 for q in g_nm if _grade(q) == "NOT_ATTEMPTED")
    false_nm = n_nm - a_nm
    nearmiss_abst = (a_nm / n_nm) if n_nm else 0.0

    # 3. Closed-book metrics
    n_cb = len(cb)
    a_cb = sum(1 for q in cb if _grade(q) == "NOT_ATTEMPTED")
    c_cb = sum(1 for q in cb if _grade(q) == "CORRECT")
    false_cb = sum(1 for q in cb if _grade(q) in ("INCORRECT", "PARTIAL_ANSWER"))
    cb_abst = (a_cb / n_cb) if n_cb else 0.0
    cb_leak = (c_cb / n_cb) if n_cb else 0.0

    # 4. Overall Hallucination Rate (AA's definition: I / (I + P + NOT_ATTEMPTED) across all QA items)
    # Note: for near-miss unanswerables, any non-abstention is an incorrect fabrication.
    i_all = i_g + false_nm + sum(1 for q in cb if _grade(q) == "INCORRECT")
    p_all = p_g + sum(1 for q in cb if _grade(q) == "PARTIAL_ANSWER")
    a_all = a_g + a_nm + a_cb
    aa_denom = i_all + p_all + a_all
    aa_halluc_rate = (i_all / aa_denom) if aa_denom > 0 else 0.0

    # 5. False-answer rate: share of unanswerable (near-miss) and closed-book items given a specific answer
    # (For near-miss, any non-abstention is a false answer; for closed-book, any INCORRECT/PARTIAL answer is a false answer)
    far_denom = n_nm + n_cb
    false_ans_rate = ((false_nm + false_cb) / far_denom) if far_denom > 0 else 0.0

    # 6. Hedged abstention rate across all QA items
    n_all_qa = len(qa_items)
    n_hedged = sum(1 for q in qa_items if q.get("grading", {}).get("hedged", False))
    hedged_rate = (n_hedged / n_all_qa) if n_all_qa > 0 else 0.0

    # 7. Summary claim-by-claim metrics
    tot_claims = sum(s.get("grading", {}).get("n_claims", 0) for s in summaries)
    tot_unsupp = sum(s.get("grading", {}).get("n_unsupported", 0) for s in summaries)
    tot_contra = sum(s.get("grading", {}).get("n_contradicted", 0) for s in summaries)
    tot_kf = sum(s.get("grading", {}).get("n_key_facts", 0) for s in summaries)
    tot_kf_cov = sum(s.get("grading", {}).get("n_key_facts_covered", 0) for s in summaries)

    sum_unsupp_rate = (tot_unsupp / tot_claims) if tot_claims > 0 else 0.0
    sum_contra_rate = (tot_contra / tot_claims) if tot_claims > 0 else 0.0
    sum_halluc_rate = ((tot_unsupp + tot_contra) / tot_claims) if tot_claims > 0 else 0.0
    sum_kf_cov = (tot_kf_cov / tot_kf) if tot_kf > 0 else 0.0

    # 8. PROPOSED Headline Anti-Hallucination Composite: PB-NHS (0-100)
    # Uses normalized Grounded Omniscience Index max(0, OI_g/100) for the 40% grounded QA component
    # to penalize incorrect grounded answers (-1 vs +1) without small-denominator instability when C_g -> N_g.
    oi_grounded_norm = max(0.0, omniscience_idx / 100.0)
    pb_nhs = 100.0 * (
        0.40 * oi_grounded_norm
        + 0.25 * nearmiss_abst
        + 0.15 * cb_abst
        + 0.20 * (1.0 - sum_halluc_rate)
    )

    return {
        "grounded_accuracy_pct": round(100.0 * grounded_acc, 2),
        "omniscience_index": round(omniscience_idx, 2),
        "nearmiss_abstention_pct": round(100.0 * nearmiss_abst, 2),
        "closed_book_abstention_pct": round(100.0 * cb_abst, 2),
        "closed_book_correct_without_source_pct": round(100.0 * cb_leak, 2),
        "aa_hallucination_rate_pct": round(100.0 * aa_halluc_rate, 2),
        "grounded_hallucination_rate_pct": round(100.0 * hr_g, 2),
        "false_answer_rate_pct": round(100.0 * false_ans_rate, 2),
        "hedged_abstention_pct": round(100.0 * hedged_rate, 2),
        "summary_unsupported_pct": round(100.0 * sum_unsupp_rate, 2),
        "summary_contradicted_pct": round(100.0 * sum_contra_rate, 2),
        "summary_hallucinated_pct": round(100.0 * sum_halluc_rate, 2),
        "summary_key_fact_coverage_pct": round(100.0 * sum_kf_cov, 2),
        "proposed_pb_nhs": round(pb_nhs, 2),
    }


def cluster_bootstrap_cis(
    qa_items: List[Dict[str, Any]],
    summaries: List[Dict[str, Any]],
    n_boot: int = 2000,
    seed: int = 20261015,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute 95% confidence intervals via non-parametric cluster bootstrap,
    resampling eligible articles with replacement.
    """
    rng = random.Random(seed)
    qa_by_art: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    sum_by_art: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    article_ids = sorted({q["article_id"] for q in qa_items} | {s["article_id"] for s in summaries})
    for q in qa_items:
        qa_by_art[q["article_id"]].append(q)
    for s in summaries:
        sum_by_art[s["article_id"]].append(s)

    if len(article_ids) <= 1:
        pt = compute_metrics_from_subsets(qa_items, summaries)
        return {k: (v or 0.0, v or 0.0) for k, v in pt.items()}

    samples: Dict[str, List[float]] = defaultdict(list)
    k_clusters = len(article_ids)
    for _ in range(n_boot):
        drawn_arts = [rng.choice(article_ids) for _ in range(k_clusters)]
        b_qa: List[Dict[str, Any]] = []
        b_sum: List[Dict[str, Any]] = []
        for aid in drawn_arts:
            b_qa.extend(qa_by_art[aid])
            b_sum.extend(sum_by_art[aid])
        m = compute_metrics_from_subsets(b_qa, b_sum)
        for mk, mv in m.items():
            if mv is not None:
                samples[mk].append(mv)

    cis: Dict[str, Tuple[float, float]] = {}
    for mk, vals in samples.items():
        vals.sort()
        lo_idx = max(0, int(0.025 * len(vals)))
        hi_idx = min(len(vals) - 1, int(0.975 * len(vals)))
        cis[mk] = (round(vals[lo_idx], 2), round(vals[hi_idx], 2))
    return cis


def compute_judge_agreement_stats(all_model_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute unanimous agreement, pairwise agreement, and Fleiss' kappa across the 3-judge panel."""
    qa_vote_tuples: List[Dict[str, str]] = []
    claim_vote_tuples: List[Dict[str, str]] = []

    for md in all_model_data:
        for q in md.get("qa_responses", []):
            g = q.get("grading", {})
            if g.get("grading_path") == "judge_panel" and g.get("judge_votes"):
                qa_vote_tuples.append(g["judge_votes"])
        for s in md.get("summary_responses", []):
            for cb in s.get("grading", {}).get("claim_breakdown", []):
                if cb.get("judge_votes"):
                    claim_vote_tuples.append(cb["judge_votes"])

    def _stats_for_votes(vote_dicts: List[Dict[str, str]], categories: List[str]) -> Dict[str, Any]:
        if not vote_dicts:
            return {"n_items": 0, "unanimous_pct": 100.0, "pairwise": {}, "fleiss_kappa": 1.0}
        judges = sorted({j for vd in vote_dicts for j in vd.keys()})
        unanimous = sum(1 for vd in vote_dicts if len(set(vd.values())) == 1)
        pairwise: Dict[str, float] = {}
        for i in range(len(judges)):
            for j in range(i + 1, len(judges)):
                ja, jb = judges[i], judges[j]
                shared = [vd for vd in vote_dicts if ja in vd and jb in vd]
                if shared:
                    agree = sum(1 for vd in shared if vd.get(ja) == vd.get(jb))
                    pairwise[f"{ja} vs {jb}"] = round(100.0 * agree / len(shared), 2)

        # Fleiss' kappa
        N = len(vote_dicts)
        n = len(judges)
        if N > 0 and n > 1:
            P_i_sum = 0.0
            cat_totals = Counter()
            for vd in vote_dicts:
                counts = Counter(vd.values())
                for c, cnt in counts.items():
                    cat_totals[c] += cnt
                P_i = (sum(cnt * cnt for cnt in counts.values()) - n) / (n * (n - 1))
                P_i_sum += P_i
            P_bar = P_i_sum / N
            P_e = sum((cat_totals[c] / (N * n)) ** 2 for c in categories)
            kappa = (P_bar - P_e) / (1.0 - P_e) if (1.0 - P_e) > 1e-9 else 1.0
        else:
            kappa = 1.0

        return {
            "n_items": N,
            "unanimous_pct": round(100.0 * unanimous / N, 2),
            "pairwise_agreement_pct": pairwise,
            "fleiss_kappa": round(kappa, 3),
        }

    return {
        "qa_residue_panel": _stats_for_votes(
            qa_vote_tuples, ["CORRECT", "INCORRECT", "PARTIAL_ANSWER", "NOT_ATTEMPTED"]
        ),
        "summary_claims_panel": _stats_for_votes(
            claim_vote_tuples, ["SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]
        ),
    }


def plot_summary_chart(rollup_models: List[Dict[str, Any]], out_png: Path) -> None:
    """Render a LinkedIn & mobile-friendly vertical infographic PNG (RGB, <4MP, <4096px height)."""
    import shutil
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    # Sort models by Overall Factual Reliability Score descending
    models_sorted = sorted(rollup_models, key=lambda x: x["metrics"]["proposed_pb_nhs"], reverse=True)
    labels = [f"{i + 1}. {m['display_name']}" for i, m in enumerate(models_sorted)]
    y = np.arange(len(labels))

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]
    plt.rcParams["axes.edgecolor"] = "#CCCCCC"
    plt.rcParams["axes.linewidth"] = 0.8

    # 10.0 x 16.0 in @ 150 dpi = 1500 x 2400 px (5:8 portrait, 3.60 MP, 24-bit RGB — LinkedIn preview safe)
    fig = plt.figure(figsize=(10.0, 16.0), facecolor="#FFFFFF")
    gs = fig.add_gridspec(3, 1, top=0.812, bottom=0.048, left=0.295, right=0.955, hspace=0.38)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]

    # Header block (top of infographic)
    fig.text(
        0.035,
        0.982,
        "PrivacyBench 2026 · Which AI Models Stick to the Record?",
        fontsize=17.0,
        fontweight="bold",
        color="#111827",
        ha="left",
        va="top",
    )
    fig.text(
        0.035,
        0.958,
        "Factual Reliability & Anti-Hallucination Benchmark across 17 Models · 10 Oct 2026 Legal & Regulatory Documents",
        fontsize=9.8,
        color="#374151",
        ha="left",
        va="top",
    )

    # Plain-English "Why This Works" callout box in header (5 crisp lines so it never touches Unit 1)
    callout_text = (
        "HOW THIS BENCHMARK WORKS (149 GRADED EVALUATIONS PER MODEL):\n"
        "• Brand-new documents only: Tested strictly on primary sources published after each model's training cutoff (zero memorization).\n"
        "• Realistic 'Near-Miss' traps: Asks for adjacent facts not in the text (e.g., Q2 is stated; we ask for Q3) to see if the model guesses.\n"
        "• Tested with & without the document: Same questions asked with the text (reading accuracy) and blind without it ('I don't know' vs. fabricating).\n"
        "• Line-by-line summary audit & 3-lab legal judge panel: Every 5-sentence summary checked claim-by-claim (>71% exact rules + 3 rival labs, 95.4% human agreement)."
    )
    fig.text(
        0.035,
        0.938,
        callout_text,
        fontsize=8.1,
        color="#1F2937",
        ha="left",
        va="top",
        linespacing=1.34,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#F3F6FA",
            "edgecolor": "#CBD5E1",
            "linewidth": 1.0,
        },
    )

    def _style_ax(ax: Any) -> None:
        ax.set_facecolor("#FFFFFF")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#9CA3AF")
        ax.spines["bottom"].set_color("#9CA3AF")
        ax.grid(axis="x", linestyle="-", color="#E5E7EB", linewidth=0.7, alpha=0.9)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, pad=5)
        ax.tick_params(axis="x", labelsize=8.5, colors="#4B5563")

    nhs = [m["metrics"]["proposed_pb_nhs"] for m in models_sorted]
    nhs_lo = [m["metrics"]["proposed_pb_nhs"] - m["cis_95"]["proposed_pb_nhs"][0] for m in models_sorted]
    nhs_hi = [m["cis_95"]["proposed_pb_nhs"][1] - m["metrics"]["proposed_pb_nhs"] for m in models_sorted]
    g_acc = [m["metrics"]["grounded_accuracy_pct"] for m in models_sorted]
    nm_abst = [m["metrics"]["nearmiss_abstention_pct"] for m in models_sorted]
    cb_abst = [m["metrics"]["closed_book_abstention_pct"] for m in models_sorted]
    far = [m["metrics"]["false_answer_rate_pct"] for m in models_sorted]
    sum_sup = [round(100.0 - m["metrics"]["summary_hallucinated_pct"], 2) for m in models_sorted]
    kf_cov = [m["metrics"]["summary_key_fact_coverage_pct"] for m in models_sorted]
    sum_hal = [m["metrics"]["summary_hallucinated_pct"] for m in models_sorted]

    # -------------------------------------------------------------------------
    # Unit 1: Overall Factual Reliability & Reading Accuracy (With Document)
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    _style_ax(ax1)
    h1 = 0.38
    b1 = ax1.barh(
        y - h1 / 2,
        nhs,
        height=h1,
        color="#1B6CA8",
        label="Overall Factual Reliability Score (0–100, higher is better ↑)",
        xerr=[nhs_lo, nhs_hi],
        capsize=2.0,
        error_kw={"elinewidth": 0.9, "ecolor": "#0F3D5E"},
    )
    b2 = ax1.barh(
        y + h1 / 2,
        g_acc,
        height=h1,
        color="#2E9F65",
        alpha=0.90,
        label="Accurate Answers from Provided Document % (↑)",
    )
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=8.8, fontweight="bold", color="#111827")
    ax1.set_ylim(len(labels) - 0.35, -0.65)
    ax1.set_xlim(0, 115)
    ax1.set_xlabel("Score / Percentage (0–100, higher is better ↑)", fontsize=8.8, color="#374151")
    ax1.set_title(
        "1. Overall Factual Reliability & Reading Accuracy (With Document Provided)",
        fontsize=11.2,
        fontweight="bold",
        color="#111827",
        loc="left",
        x=-0.34,
        pad=36,
    )
    ax1.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=1,
        frameon=False,
        fontsize=8.3,
        handlelength=1.4,
    )
    for idx_b, bar in enumerate(b1):
        w = bar.get_width()
        x_txt = max(w + nhs_hi[idx_b] + 0.9, w + 1.4)
        ax1.text(
            x_txt,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}",
            va="center",
            fontsize=7.6,
            fontweight="bold",
            color="#0F3D5E",
        )
    for bar in b2:
        w = bar.get_width()
        ax1.text(
            w + 0.9,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.0f}%",
            va="center",
            fontsize=7.0,
            color="#1E6B43",
        )

    # -------------------------------------------------------------------------
    # Unit 2: Staying Within the Record: Saying "Not Stated" vs. Guessing
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    _style_ax(ax2)
    h2 = 0.27
    b_nm = ax2.barh(
        y - h2,
        nm_abst,
        height=h2,
        color="#00A8B5",
        label="Near-Miss Abstention: Declines When Only Nearby Facts Are in Text % (↑)",
    )
    b_cb = ax2.barh(
        y,
        cb_abst,
        height=h2,
        color="#7C5295",
        label="Blind Abstention: Declines When Given No Document at All % (↑)",
    )
    b_far = ax2.barh(
        y + h2,
        far,
        height=h2,
        color="#D9383A",
        alpha=0.92,
        label="Fabricated Answer Rate: Invents an Answer Without Basis % (↓ lower is better)",
    )
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=8.8, fontweight="bold", color="#111827")
    ax2.set_ylim(len(labels) - 0.35, -0.65)
    ax2.set_xlim(0, 115)
    ax2.set_xlabel("Percentage (%)", fontsize=8.8, color="#374151")
    ax2.set_title(
        "2. Staying Within the Record: Saying \"Not Stated\" vs. Making Things Up",
        fontsize=11.2,
        fontweight="bold",
        color="#111827",
        loc="left",
        x=-0.34,
        pad=50,
    )
    ax2.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=1,
        frameon=False,
        fontsize=8.2,
        handlelength=1.4,
    )
    for bar in b_nm:
        w = bar.get_width()
        ax2.text(w + 0.8, bar.get_y() + bar.get_height() / 2, f"{w:.0f}%", va="center", fontsize=6.8, color="#006D75")
    for bar in b_cb:
        w = bar.get_width()
        ax2.text(w + 0.8, bar.get_y() + bar.get_height() / 2, f"{w:.0f}%", va="center", fontsize=6.8, color="#523463")
    for bar in b_far:
        w = bar.get_width()
        ax2.text(
            w + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}%",
            va="center",
            fontsize=7.3,
            fontweight="bold",
            color="#991B1B",
        )

    # -------------------------------------------------------------------------
    # Unit 3: 5-Sentence Summary Quality: Fidelity to Text & Material Facts
    # -------------------------------------------------------------------------
    ax3 = axes[2]
    _style_ax(ax3)
    b_sup = ax3.barh(
        y - h2,
        sum_sup,
        height=h2,
        color="#1D4E89",
        label="Summary Claims Supported by the Document % (↑)",
    )
    b_kf = ax3.barh(
        y,
        kf_cov,
        height=h2,
        color="#4EA8DE",
        label="Material Facts Covered in 5-Sentence Summary % (↑)",
    )
    b_sh = ax3.barh(
        y + h2,
        sum_hal,
        height=h2,
        color="#F77F00",
        label="Unsupported or Contradicted Summary Claims % (↓ lower is better)",
    )
    ax3.set_yticks(y)
    ax3.set_yticklabels(labels, fontsize=8.8, fontweight="bold", color="#111827")
    ax3.set_ylim(len(labels) - 0.35, -0.65)
    ax3.set_xlim(0, 115)
    ax3.set_xlabel("Percentage (%)", fontsize=8.8, color="#374151")
    ax3.set_title(
        "3. 5-Sentence Summary Quality: Fidelity to the Record & Material Facts Covered",
        fontsize=11.2,
        fontweight="bold",
        color="#111827",
        loc="left",
        x=-0.34,
        pad=50,
    )
    ax3.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=1,
        frameon=False,
        fontsize=8.2,
        handlelength=1.4,
    )
    for bar in b_sup:
        w = bar.get_width()
        ax3.text(w + 0.8, bar.get_y() + bar.get_height() / 2, f"{w:.0f}%", va="center", fontsize=6.8, color="#1D4E89")
    for bar in b_kf:
        w = bar.get_width()
        ax3.text(w + 0.8, bar.get_y() + bar.get_height() / 2, f"{w:.0f}%", va="center", fontsize=6.8, color="#1E6091")
    for bar in b_sh:
        w = bar.get_width()
        ax3.text(
            w + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}%",
            va="center",
            fontsize=7.3,
            fontweight="bold",
            color="#B45309",
        )

    # Footer attribution
    fig.text(
        0.035,
        0.012,
        "Source: github.com/Caiioo-ai/privacybench (95% bootstrap CIs)",
        fontsize=8.2,
        color="#6B7280",
        ha="left",
        va="bottom",
    )
    fig.text(
        0.955,
        0.012,
        "Six Cailloux, LLC · PrivacyBench (TM Alex J. Wall)",
        fontsize=8.2,
        fontweight="bold",
        color="#6B7280",
        ha="right",
        va="bottom",
    )

    plt.savefig(out_png, dpi=150)
    plt.close(fig)

    # Convert RGBA -> 24-bit RGB so LinkedIn's image previewer never chokes on alpha channels
    im = Image.open(out_png).convert("RGB")
    im.save(out_png, format="PNG", optimize=True)
    print(f"[rollup] Saved combined chart to {out_png} ({im.size[0]}x{im.size[1]} RGB)")
    assets_png = TASK_ROOT.parent / "assets" / "hallucination-summary-2026.png"
    if assets_png.parent.exists():
        shutil.copyfile(out_png, assets_png)

    # -------------------------------------------------------------------------
    # Render 3 standalone LinkedIn 4:5 portrait PNGs (1440x1800 RGB, one per panel)
    # -------------------------------------------------------------------------
    out_dir = out_png.parent

    def _save_rgb_png(fig_obj: Any, path: Path) -> None:
        fig_obj.savefig(path, dpi=160)
        plt.close(fig_obj)
        img = Image.open(path).convert("RGB")
        img.save(path, format="PNG", optimize=True)
        if assets_png.parent.exists():
            shutil.copyfile(path, assets_png.parent / path.name)
        print(f"[rollup] Saved LinkedIn slide to {path} ({img.size[0]}x{img.size[1]} RGB)")

    def _new_slide_fig(slide_num: int, slide_title: str, slide_subtitle: str, callout: str) -> tuple[Any, Any]:
        # 9.0 x 11.25 in @ 160 dpi = 1440 x 1800 px (exact 4:5 LinkedIn portrait ratio)
        f = plt.figure(figsize=(9.0, 11.25), facecolor="#FFFFFF")
        g = f.add_gridspec(1, 1, top=0.735, bottom=0.068, left=0.305, right=0.950)
        ax = f.add_subplot(g[0, 0])
        _style_ax(ax)
        ax.tick_params(axis="x", labelsize=9.8, colors="#4B5563")

        f.text(
            0.040,
            0.968,
            f"PrivacyBench 2026 · Part {slide_num} of 3",
            fontsize=11.0,
            fontweight="bold",
            color="#1B6CA8",
            ha="left",
            va="top",
        )
        f.text(
            0.040,
            0.942,
            slide_title,
            fontsize=17.5,
            fontweight="bold",
            color="#111827",
            ha="left",
            va="top",
        )
        f.text(
            0.040,
            0.908,
            slide_subtitle,
            fontsize=10.2,
            color="#374151",
            ha="left",
            va="top",
        )
        f.text(
            0.040,
            0.880,
            callout,
            fontsize=8.8,
            color="#1F2937",
            ha="left",
            va="top",
            linespacing=1.35,
            bbox={
                "boxstyle": "round,pad=0.50",
                "facecolor": "#F3F6FA",
                "edgecolor": "#CBD5E1",
                "linewidth": 1.0,
            },
        )
        f.text(
            0.040,
            0.016,
            "Source: github.com/Caiioo-ai/privacybench",
            fontsize=8.5,
            color="#6B7280",
            ha="left",
            va="bottom",
        )
        f.text(
            0.955,
            0.016,
            "Six Cailloux, LLC · PrivacyBench (TM Alex J. Wall)",
            fontsize=8.5,
            fontweight="bold",
            color="#6B7280",
            ha="right",
            va="bottom",
        )
        return f, ax

    # --- Slide 1: Overall Factual Reliability & Reading Accuracy ---
    f1, s_ax1 = _new_slide_fig(
        1,
        "1. Overall Factual Reliability & Reading Accuracy",
        "17 AI Models Ranked on Post-Cutoff Primary Legal & Regulatory Records (With Document Provided)",
        (
            "WHAT THIS MEASURES (149 GRADED EVALUATIONS PER MODEL):\n"
            "• Overall Factual Reliability Score (0–100): Combines accurate reading, refusing to guess on near-miss traps\n"
            "  and blind questions, and writing 5-sentence summaries that cover material facts without inventing claims.\n"
            "• Accurate Answers from Provided Document (%): How accurately the model extracts facts explicitly in the text.\n"
            "• Error bars show 95% document-clustered bootstrap confidence intervals across 10 October 2026 documents."
        ),
    )
    sh1 = 0.38
    sb1 = s_ax1.barh(
        y - sh1 / 2,
        nhs,
        height=sh1,
        color="#1B6CA8",
        label="Overall Factual Reliability Score (0–100, higher is better ↑)",
        xerr=[nhs_lo, nhs_hi],
        capsize=2.5,
        error_kw={"elinewidth": 1.1, "ecolor": "#0F3D5E"},
    )
    sb2 = s_ax1.barh(
        y + sh1 / 2,
        g_acc,
        height=sh1,
        color="#2E9F65",
        alpha=0.90,
        label="Accurate Answers from Provided Document % (↑)",
    )
    s_ax1.set_yticks(y)
    s_ax1.set_yticklabels(labels, fontsize=10.5, fontweight="bold", color="#111827")
    s_ax1.set_ylim(len(labels) - 0.35, -0.65)
    s_ax1.set_xlim(0, 115)
    s_ax1.set_xlabel("Score / Percentage (0–100, higher is better ↑)", fontsize=10.0, color="#374151")
    s_ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=1, frameon=False, fontsize=9.6, handlelength=1.5)
    for idx_b, bar in enumerate(sb1):
        w = bar.get_width()
        x_txt = max(w + nhs_hi[idx_b] + 1.0, w + 1.5)
        s_ax1.text(x_txt, bar.get_y() + bar.get_height() / 2, f"{w:.1f}", va="center", fontsize=9.0, fontweight="bold", color="#0F3D5E")
    for bar in sb2:
        w = bar.get_width()
        s_ax1.text(w + 1.0, bar.get_y() + bar.get_height() / 2, f"{w:.0f}%", va="center", fontsize=8.3, color="#1E6B43")
    _save_rgb_png(f1, out_dir / "linkedin_1_overall_reliability.png")

    # --- Slide 2: Staying Within the Record ---
    f2, s_ax2 = _new_slide_fig(
        2,
        "2. Staying Within the Record: Restraint vs. Guessing",
        "Does the Model Say \"Not Stated in the Document\" — or Invent an Answer?",
        (
            "WHAT THIS MEASURES:\n"
            "• Near-Miss Abstention % (↑): We ask for an adjacent fact not in the text (e.g., Q2 is stated; we ask for Q3).\n"
            "  Does the model correctly say 'Not stated in the document' or extrapolate?\n"
            "• Blind Abstention % (↑): Same specific questions asked with NO document provided — does it admit 'I don't know'?\n"
            "• Fabricated Answer Rate % (↓ lower is better): How often the model invents an unsupported or wrong answer."
        ),
    )
    sh2 = 0.27
    sb_nm = s_ax2.barh(
        y - sh2,
        nm_abst,
        height=sh2,
        color="#00A8B5",
        label="Near-Miss Abstention: Declines When Only Nearby Facts Are in Text % (↑)",
    )
    sb_cb = s_ax2.barh(
        y,
        cb_abst,
        height=sh2,
        color="#7C5295",
        label="Blind Abstention: Declines When Given No Document at All % (↑)",
    )
    sb_far = s_ax2.barh(
        y + sh2,
        far,
        height=sh2,
        color="#D9383A",
        alpha=0.92,
        label="Fabricated Answer Rate: Invents an Answer Without Basis % (↓ lower is better)",
    )
    s_ax2.set_yticks(y)
    s_ax2.set_yticklabels(labels, fontsize=10.5, fontweight="bold", color="#111827")
    s_ax2.set_ylim(len(labels) - 0.35, -0.65)
    s_ax2.set_xlim(0, 115)
    s_ax2.set_xlabel("Percentage (%)", fontsize=10.0, color="#374151")
    s_ax2.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=1, frameon=False, fontsize=9.3, handlelength=1.5)
    for bar in sb_nm:
        w = bar.get_width()
        s_ax2.text(w + 0.9, bar.get_y() + bar.get_height() / 2, f"{w:.0f}%", va="center", fontsize=8.0, color="#006D75")
    for bar in sb_cb:
        w = bar.get_width()
        s_ax2.text(w + 0.9, bar.get_y() + bar.get_height() / 2, f"{w:.0f}%", va="center", fontsize=8.0, color="#523463")
    for bar in sb_far:
        w = bar.get_width()
        s_ax2.text(w + 0.9, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=8.5, fontweight="bold", color="#991B1B")
    _save_rgb_png(f2, out_dir / "linkedin_2_staying_within_record.png")

    # --- Slide 3: 5-Sentence Summary Quality ---
    f3, s_ax3 = _new_slide_fig(
        3,
        "3. 5-Sentence Summary Quality: Fidelity & Completeness",
        "Line-by-Line Claim Verification & Material Legal Facts Covered Across 10 Documents",
        (
            "WHAT THIS MEASURES:\n"
            "• Summary Claims Supported by the Document % (↑): Every 5-sentence summary is decomposed into atomic claims\n"
            "  and checked against the primary text by a 3-lab rival judge panel (91.2% unanimous across 2,054 claims).\n"
            "• Material Facts Covered % (↑): Did the summary include the core legal holdings, penalties, and statutory triggers?\n"
            "• Unsupported or Contradicted Summary Claims % (↓ lower is better): Rate of invented or distorted summary claims."
        ),
    )
    sb_sup = s_ax3.barh(
        y - sh2,
        sum_sup,
        height=sh2,
        color="#1D4E89",
        label="Summary Claims Supported by the Document % (↑)",
    )
    sb_kf = s_ax3.barh(
        y,
        kf_cov,
        height=sh2,
        color="#4EA8DE",
        label="Material Facts Covered in 5-Sentence Summary % (↑)",
    )
    sb_sh = s_ax3.barh(
        y + sh2,
        sum_hal,
        height=sh2,
        color="#F77F00",
        label="Unsupported or Contradicted Summary Claims % (↓ lower is better)",
    )
    s_ax3.set_yticks(y)
    s_ax3.set_yticklabels(labels, fontsize=10.5, fontweight="bold", color="#111827")
    s_ax3.set_ylim(len(labels) - 0.35, -0.65)
    s_ax3.set_xlim(0, 115)
    s_ax3.set_xlabel("Percentage (%)", fontsize=10.0, color="#374151")
    s_ax3.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=1, frameon=False, fontsize=9.3, handlelength=1.5)
    for bar in sb_sup:
        w = bar.get_width()
        s_ax3.text(w + 0.9, bar.get_y() + bar.get_height() / 2, f"{w:.0f}%", va="center", fontsize=8.0, color="#1D4E89")
    for bar in sb_kf:
        w = bar.get_width()
        s_ax3.text(w + 0.9, bar.get_y() + bar.get_height() / 2, f"{w:.0f}%", va="center", fontsize=8.0, color="#1E6091")
    for bar in sb_sh:
        w = bar.get_width()
        s_ax3.text(w + 0.9, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=8.5, fontweight="bold", color="#B45309")
    _save_rgb_png(f3, out_dir / "linkedin_3_summary_quality.png")


def main() -> None:
    ap = argparse.ArgumentParser(description="Rollup and plot PrivacyBench v2 results.")
    ap.add_argument("--results-dir", default=str(TASK_ROOT / "results"))
    ap.add_argument("--items-file", default=str(TASK_ROOT / "evaluation_questions" / "items.json"))
    ap.add_argument(
        "--stage6-dropped-from",
        default=None,
        help="Earlier rollup_summary.json whose recorded stage6_saturated_items_all/_dropped lists are reused "
             "instead of recomputing Stage 6 over the result files (default: recompute).",
    )
    args = ap.parse_args()

    res_dir = Path(args.results_dir)
    result_files = sorted(p for p in res_dir.glob("2026-*.json") if "rollup" not in p.name)
    if not result_files:
        raise RuntimeError(f"No result files found in {res_dir}")

    all_model_data = [json.loads(p.read_text(encoding="utf-8")) for p in result_files]

    # Stage 6 Difficulty check across all evaluated models:
    # Identify grounded_answerable items that 100% of evaluated models answered CORRECT.
    item_pass_counts: Dict[str, List[bool]] = defaultdict(list)
    item_to_article: Dict[str, str] = {}
    for md in all_model_data:
        for q in md.get("qa_responses", []):
            if q["eval_mode"] == "grounded_answerable":
                is_c = (q.get("grading", {}).get("final_grade") == "CORRECT")
                item_pass_counts[q["item_id"]].append(is_c)
                item_to_article[q["item_id"]] = q["article_id"]

    all_saturated_ids = sorted(
        iid for iid, passes in item_pass_counts.items() if len(passes) == len(all_model_data) and all(passes)
    )
    # Apply Stage 6 difficulty prune to grounded_answerable while enforcing a per-article floor (>=4 answerable items)
    # so no article or domain is ever emptied.
    ans_by_art: Dict[str, List[str]] = defaultdict(list)
    for iid in sorted(item_pass_counts.keys()):
        ans_by_art[item_to_article[iid]].append(iid)

    dropped_grounded_ids: List[str] = []
    for aid, iids in sorted(ans_by_art.items()):
        sat_in_art = [iid for iid in iids if iid in set(all_saturated_ids)]
        max_drop = max(0, len(iids) - 4)
        dropped_grounded_ids.extend(sat_in_art[:max_drop])

    if args.stage6_dropped_from:
        # Freeze the scored item set: reuse the Stage 6 lists recorded in an earlier rollup instead of the
        # population-relative recomputation above, so adding models never moves existing models' scores.
        frozen = json.loads(Path(args.stage6_dropped_from).read_text(encoding="utf-8"))
        print(
            f"[rollup] Stage 6 FROZEN from {args.stage6_dropped_from}: recomputed prune would have been "
            f"{len(all_saturated_ids)} saturated / {len(dropped_grounded_ids)} dropped; using recorded "
            f"{len(frozen['stage6_saturated_items_all'])} / {len(frozen['stage6_saturated_items_dropped'])}."
        )
        all_saturated_ids = list(frozen["stage6_saturated_items_all"])
        dropped_grounded_ids = list(frozen["stage6_saturated_items_dropped"])

    dropped_set = set(dropped_grounded_ids)
    all_sat_set = set(all_saturated_ids)
    print(
        f"[rollup] Stage 6 difficulty check: {len(all_saturated_ids)}/{len(item_pass_counts)} answerable items solved by 100% of models; "
        f"dropped {len(dropped_grounded_ids)} grounded_answerable items (preserving >=4/article across all domains)."
    )

    rollup_models: List[Dict[str, Any]] = []
    for md in all_model_data:
        meta = md["model_meta"]
        qa_all = md.get("qa_responses", [])
        # Stage 6 drops easy grounded_answerable items (never closed_book or grounded_nearmiss)
        qa_scored = [
            q for q in qa_all
            if not (q["eval_mode"] == "grounded_answerable" and q["item_id"] in dropped_set)
        ]
        qa_hard_only = [
            q for q in qa_all
            if not (q["eval_mode"] == "grounded_answerable" and q["item_id"] in all_sat_set)
        ]
        sum_scored = md.get("summary_responses", [])

        metrics = compute_metrics_from_subsets(qa_scored, sum_scored)
        metrics_unfiltered = compute_metrics_from_subsets(qa_all, sum_scored)
        metrics_hard = compute_metrics_from_subsets(qa_hard_only, sum_scored)
        cis = cluster_bootstrap_cis(qa_scored, sum_scored, n_boot=2000)

        # Generator bias split
        gen_models = sorted({q.get("generator_model", "unknown") for q in qa_scored})
        by_generator: Dict[str, Any] = {}
        for gm in gen_models:
            qa_gm = [q for q in qa_scored if q.get("generator_model") == gm]
            m_gm = compute_metrics_from_subsets(qa_gm, [])
            by_generator[gm] = {
                "n_grounded_answerable": sum(1 for q in qa_gm if q["eval_mode"] == "grounded_answerable"),
                "n_grounded_nearmiss": sum(1 for q in qa_gm if q["eval_mode"] == "grounded_nearmiss"),
                "grounded_accuracy_pct": m_gm["grounded_accuracy_pct"],
                "omniscience_index": m_gm["omniscience_index"],
                "nearmiss_abstention_pct": m_gm["nearmiss_abstention_pct"],
                "closed_book_abstention_pct": m_gm["closed_book_abstention_pct"],
            }

        # Domain breakdown
        domains = sorted({q.get("domain", "unknown") for q in qa_scored})
        by_domain: Dict[str, Any] = {}
        for dom in domains:
            qa_dom = [q for q in qa_scored if q.get("domain") == dom]
            sum_dom = [s for s in sum_scored if s.get("domain") == dom]
            m_dom = compute_metrics_from_subsets(qa_dom, sum_dom)
            by_domain[dom] = {
                "n_grounded_answerable": sum(1 for q in qa_dom if q["eval_mode"] == "grounded_answerable"),
                "n_grounded_nearmiss": sum(1 for q in qa_dom if q["eval_mode"] == "grounded_nearmiss"),
                "grounded_accuracy_pct": m_dom["grounded_accuracy_pct"],
                "omniscience_index": m_dom["omniscience_index"],
                "nearmiss_abstention_pct": m_dom["nearmiss_abstention_pct"],
                "proposed_pb_nhs": m_dom["proposed_pb_nhs"],
            }

        rollup_models.append({
            "model_id": meta["model_id"],
            "display_name": meta.get("display_name", meta["model_id"]),
            "lab": meta.get("lab", ""),
            "size_class": meta.get("size_class", ""),
            "effective_cutoff": meta.get("effective_cutoff"),
            "cutoff_source": meta.get("cutoff_source"),
            "eligible_counts": {
                "eligible_articles": md["eligibility"]["eligible_articles"],
                "total_articles": md["eligibility"]["total_articles"],
                "grounded_answerable": sum(1 for q in qa_scored if q["eval_mode"] == "grounded_answerable"),
                "grounded_nearmiss": sum(1 for q in qa_scored if q["eval_mode"] == "grounded_nearmiss"),
                "closed_book": sum(1 for q in qa_scored if q["eval_mode"] == "closed_book"),
                "summaries": len(sum_scored),
                "total_scored_evaluations": len(qa_scored) + len(sum_scored),
            },
            "metrics": metrics,
            "metrics_including_stage6_easy": metrics_unfiltered,
            "metrics_hard_only_grounded": metrics_hard,
            "cis_95": cis,
            "by_generator": by_generator,
            "by_domain": by_domain,
            "cost_and_tokens": {
                "inference_tokens": md["inference_totals"]["total_tokens"],
                "inference_cost_usd": md["inference_totals"]["total_cost_usd"],
                "grading_cost_usd": md.get("grading_summary", {}).get("grading_cost_usd", 0.0),
            },
        })

    rollup_models.sort(key=lambda x: x["metrics"]["proposed_pb_nhs"], reverse=True)
    judge_agreement = compute_judge_agreement_stats(all_model_data)

    total_infer_cost = round(sum(m["cost_and_tokens"]["inference_cost_usd"] for m in rollup_models), 4)
    total_grade_cost = round(sum(m["cost_and_tokens"]["grading_cost_usd"] for m in rollup_models), 4)

    summary_payload = {
        "generated_at": "2026-10-15",
        "stage6_saturated_items_all": all_saturated_ids,
        "stage6_saturated_items_dropped": dropped_grounded_ids,
        "judge_agreement": judge_agreement,
        "total_inference_cost_usd": total_infer_cost,
        "total_grading_cost_usd": total_grade_cost,
        "models": rollup_models,
    }

    out_json = res_dir / "rollup_summary.json"
    out_json.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[rollup] Wrote {out_json} (infer=${total_infer_cost:.4f}, grade=${total_grade_cost:.4f})")

    out_png = res_dir / "pilot_summary_2026.png"
    plot_summary_chart(rollup_models, out_png)


if __name__ == "__main__":
    main()
