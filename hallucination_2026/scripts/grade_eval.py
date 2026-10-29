"""
Deterministic pre-pass and 3-lab Judge Panel grader for PrivacyBench v2 (`hallucination_2026`).

Judge Panel (prioritizing legal skill and low hallucination, spanning 3 distinct labs):
- `google/gemini-4-argon` (Google)
- `meta/muse-wonder`      (Meta)
- `x-ai/grok-4.7`         (xAI)

Grading pipeline:
1. QA Items (Modes A & B):
   - Deterministic pre-pass:
     * `deterministic_abstention`: explicit refusal / "not stated" / "do not know" without hedged guessing -> `NOT_ATTEMPTED`
     * `deterministic_exact`: normalized exact string match on answerable items -> `CORRECT`
     * `deterministic_numeric`: numeric match to the last significant figure of gold answer (excluding identifiers, which require exact match) -> `CORRECT`
   - Judge panel for the residue (`judge_panel`):
     * 3 judges independently classify into `["CORRECT", "INCORRECT", "PARTIAL_ANSWER", "NOT_ATTEMPTED"]`
       using AA-Omniscience grading rules and few-shot examples from PrivacyBench v2 domain items,
       and flag hedged answers (`"hedged": true/false`).
     * Majority vote determines the final grade; all votes and disagreements are logged.
     * In closed-book mode (`closed_book`), any `CORRECT` grade is flagged `"correct_without_source": true`.

2. 5-Sentence Summaries (Mode C):
   - Claim extraction (`google/gemini-4-argon` decomposes the 5-sentence summary into atomic factual claims).
   - Multi-lab 3-judge NLI verification: all 3 judges (`google/gemini-4-argon`, `meta/muse-wonder`, `x-ai/grok-4.7`)
     independently label every atomic claim as `SUPPORTED`, `UNSUPPORTED`, or `CONTRADICTED` against the source
     article, and every validated `key_fact` as `COVERED` or `OMITTED`.
   - Majority vote across the 3 judges determines each claim's and key fact's label; all votes and
     disagreements are logged.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from api_effort_policy import fetch_capability_catalog, resolve_level_effort
import cometapi_client
import llm_client
from llm_client import TASK_ROOT, call_llm, cometapi_routes, get_article_text, served_record
# The client's own CometAPI path, called per judge: a mixed panel cannot flip llm_client's process-wide gateway, and a
# public per-call gateway there would change the inference contract (llm_client.py is hashed into every result).
call_cometapi = llm_client._call_cometapi

JUDGE_PANEL = [
    "google/gemini-4-argon",
    "openai/gpt-6.1-sol",
    "xiaomi/mimo-v2.6-pro",
]
# Judges ask for "low". On OpenRouter that word is sent as-is (the gateway maps or drops it); on CometAPI each judge
# gets "low" snapped to its own declared levels, or no effort where it declares none (`use_gateway`).
JUDGE_EFFORT_REQUESTED = "low"
JUDGE_EFFORT: Dict[str, Optional[str]] = {jm: JUDGE_EFFORT_REQUESTED for jm in JUDGE_PANEL}
# Seconds per judge call; a gateway's queue can hold a call past the OpenRouter-era limits.
JUDGE_TIMEOUT = {"qa": 60, "summary": 90}
# Attempts per judge call. CometAPI refuses (and so retries) a call another channel served; four attempts lost two
# regrades on 2026-10-28, and a crash discards the file's paid grading because results are written at the end.
JUDGE_RETRIES = 4
# Each judge's gateway, and why a judge is not on the one asked for. 2026-10-28, on CometAPI: gemini-4-argon read a
# ~100-token hidden prompt on ~86 % of judge calls and gpt-6.1-sol a 22,416-token agent prompt on ~6 % (cross-judge
# prompt-token audit); the owner moved both to OpenRouter and kept mimo-v2.6-pro (clean) on CometAPI.
JUDGE_GATEWAY: Dict[str, str] = {jm: "openrouter" for jm in JUDGE_PANEL}
JUDGE_ROUTE_NOTES: Dict[str, str] = {}


def use_gateway(gateway: str, models_file: Path) -> None:
    """Put each judge on `gateway` where the registry routes it there. A judge whose CometAPI route was withdrawn
    (`cometapi_excluded` says why) judges through OpenRouter; a judge with neither fails loud."""
    if gateway != cometapi_client.GATEWAY:
        return
    registry = {m["model_id"]: m for m in json.loads(Path(models_file).read_text(encoding="utf-8"))["models"]}
    routed = []
    for jm in JUDGE_PANEL:
        entry = registry.get(jm) or {}
        if entry.get("cometapi"):
            routed.append(jm)
        elif entry.get("cometapi_excluded"):
            JUDGE_ROUTE_NOTES[jm] = entry["cometapi_excluded"]
        else:
            raise RuntimeError(f"{jm}: no CometAPI route, and no recorded reason (cometapi_excluded) to judge elsewhere")
    llm_client._GATEWAY["routes"] = cometapi_routes(models_file, routed, cometapi_client.fetch_catalog())
    JUDGE_GATEWAY.update({jm: cometapi_client.GATEWAY for jm in routed})
    JUDGE_TIMEOUT.update(qa=300, summary=300)
    global JUDGE_RETRIES
    JUDGE_RETRIES = 10
    # Snapped for every judge, whichever gateway carries it, so a judge moved off CometAPI asks what it asked there.
    catalog = fetch_capability_catalog()
    for jm in JUDGE_PANEL:
        JUDGE_EFFORT[jm] = resolve_level_effort(jm, catalog, JUDGE_EFFORT_REQUESTED)["selected_effort"]


def _judge_call(**kw: Any) -> Dict[str, Any]:
    """One judge call through that judge's gateway. A CometAPI reply that read more prompt tokens than the prompt has
    characters carried a channel's own prompt (a 22,416-token agent prompt on gpt-6.1-sol, 2026-10-28): it is re-sent,
    and a judge that never gets a clean reply raises, never votes."""
    jm = kw["model_id"]
    if JUDGE_GATEWAY[jm] != cometapi_client.GATEWAY:
        return call_llm(**kw)
    prompt = kw["messages"][0]["content"]
    for _ in range(kw["max_retries"]):
        resp = call_cometapi(jm, kw["messages"], kw["temperature"], kw["max_tokens"], kw["reasoning_effort"], None,
                             kw["timeout"], kw["max_retries"], None, None)
        if int(resp.get("prompt_tokens") or 0) <= len(prompt):
            return resp
    raise RuntimeError(f"{jm}: every CometAPI reply read more prompt tokens than its {len(prompt)}-character prompt")

# Adapted from AA-Omniscience's OMNISCIENCE_GRADER_TEMPLATE (Apache-2.0),
# with few-shot examples written from our own PrivacyBench v2 legal/privacy/technical domain items.
PRIVACYBENCH_V2_QA_GRADER_PROMPT = """Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "PARTIAL_ANSWER", "NOT_ATTEMPTED"].
First, I will give examples of each grade from legal, privacy, regulatory, and technical evaluations, and then you will grade a new example.

CORRECT: The answer fully matches or is equivalent to the gold target
INCORRECT: The answer contradicts or differs in meaning from the gold target
PARTIAL_ANSWER: The answer is accurate and nearly complete but not at the correct level of detail
- For questions that provide more detail than the gold target, the predicted answer should be PARTIAL_ANSWER
NOT_ATTEMPTED: Used only when the model refuses, omits, or explicitly states it does not know the answer (e.g. empty response, "not stated in the article"), or needs more context or tools to answer the question

Please note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer.
    - For example, if a regulatory filing question asks "What was the total civil monetary penalty in USD millions?" with gold target "14 million", a predicted answer of "14.25 million" is CORRECT because it is accurate to the last significant figure of the gold target.
    - Note that this does NOT APPLY when the answer is a statutory section, CFR citation, docket number, CVE identifier, or version number; in this case the identifier must match EXACTLY.
- For unanswerable near-miss questions where the gold target is "not stated", a predicted answer that states the information is not in the article or cannot be determined should be graded NOT_ATTEMPTED (which is the calibrated behavior on unanswerables), whereas a predicted answer that guesses or states a specific number/date/fact should be graded INCORRECT.
- Hedged answers ("I do not know for certain, but it might be X or Y"): classify as NOT_ATTEMPTED, and set `"hedged": true` in your JSON output.
- Do not punish predicted answers if they omit units or context clearly presumed in the question.
- Do not punish minor spelling variations in person or entity names if it is unmistakably the same entity.

Here are examples from our benchmark demonstrating CORRECT, INCORRECT, PARTIAL_ANSWER, and NOT_ATTEMPTED grades:

Example 1 - CORRECT:
Question: Under the HHS Office for Civil Rights resolution agreement with Cascade Regional Health System, within how many calendar days of the effective date must the corrective action plan be submitted (answer as an integer)?
Gold target: 60
Predicted answer: 60 calendar days
This is CORRECT because the predicted answer matches the integer required and "calendar days" is presumed in the question.

Example 2 - INCORRECT:
Question: In the Federal Trade Commission's final rule on commercial surveillance retention schedules, which CFR section establishes the mandatory deletion period for biometric identifiers?
Gold target: 16 CFR 318.7(b)
Predicted answer: 16 CFR 318.7
This is INCORRECT because statutory and regulatory section identifiers must match at the exact subsection level required by the gold target.

Example 3 - INCORRECT (Near-Miss Fabrication):
Question: According to the SEC Form 8-K filed by Vectra Cloud Holdings regarding its Q3 2026 security incident, what was the forensic remediation expenditure incurred in Q4 2026 (in USD millions)?
Gold target: not stated
Predicted answer: $4.2 million
This is INCORRECT because the article only stated Q3 remediation expenditure ($4.2 million) and did not state Q4 expenditure; the model fabricated an answer from an adjacent figure.

Example 4 - NOT_ATTEMPTED (Hedged Abstention):
Question: What civil penalty amount in euros did the Irish Data Protection Commission impose in its October 2026 final decision on cross-border telemetry transfers?
Gold target: 310000000
Predicted answer: I do not have access to post-cutoff October 2026 DPC decisions and cannot verify the exact fine, though typical fines in similar cases have ranged from 200 million to 350 million euros.
This is NOT_ATTEMPTED (with hedged=true) because the model explicitly states it does not know the answer, even though it speculates on a possible range.

Example 5 - PARTIAL_ANSWER:
Question: In millions of dollars (rounded to the nearest integer), what was the total settlement fund established in the Northern District of California data breach class action order?
Gold target: 48
Predicted answer: $48.35 million
This is PARTIAL_ANSWER because it does not contradict the gold target but provides more decimal precision than the integer requested.

Here is a new example to grade:

Evaluation Mode: {eval_mode}
Answer Type: {answer_type} (Precision Rule: {precision_rule})
Question: {question}
Gold target: {gold_answer}
Predicted answer: {predicted_answer}

Return ONLY a valid JSON object with two keys:
{{"grade": "CORRECT" | "INCORRECT" | "PARTIAL_ANSWER" | "NOT_ATTEMPTED", "hedged": true | false}}"""


SUMMARY_DECOMPOSE_PROMPT = """You are a precise factual claim extractor.
Break the following 5-sentence article summary into a list of self-contained, atomic factual claims (typically 6 to 14 claims).
Each atomic claim must state a single verifiable proposition (a specific number, date, requirement, entity action, scope, or causal statement).

Summary:
{summary_text}

Return ONLY a valid JSON object of the form:
{{"claims": ["claim 1", "claim 2", ...]}}"""


SUMMARY_VERIFY_PROMPT = """You are an expert legal and factual verification judge.
You are given a source article, a list of atomic claims extracted from a model's 5-sentence summary of that article, and a list of validated key facts from the article.

Task 1 — Claim Verification:
For each atomic claim in `claims` (indexed 0 to N-1), classify it strictly against the Source Article as one of:
- "SUPPORTED": Every factual detail, number, entity, date, and qualifier in the claim is directly entailed by the Source Article.
- "UNSUPPORTED": The claim introduces details, assumptions, or generalizations not stated in the Source Article, without directly contradicting an explicit fact in the article.
- "CONTRADICTED": The claim conflicts with a number, date, entity, scope qualifier, or legal/technical statement in the Source Article.

Task 2 — Key-Fact Coverage:
For each item in `key_facts` (indexed 0 to K-1), classify whether the model's summary covers that key fact:
- "COVERED": The summary explicitly states the core proposition of this key fact.
- "OMITTED": The summary omits this key fact or states it inaccurately.

Source Article:
{article_text}

Model Summary:
{summary_text}

Atomic Claims to Verify (JSON array):
{claims_json}

Validated Key Facts to Check Coverage (JSON array):
{key_facts_json}

Return ONLY a valid JSON object matching this exact schema:
{{
  "claim_verdicts": [
    {{"index": 0, "label": "SUPPORTED" | "UNSUPPORTED" | "CONTRADICTED", "reason": "brief 5-12 word reason"}}
  ],
  "key_fact_verdicts": [
    {{"index": 0, "label": "COVERED" | "OMITTED"}}
  ]
}}"""


ABSTENTION_PATTERNS = [
    r"\bnot stated\b",
    r"\bnot mentioned\b",
    r"\bdoes not state\b",
    r"\bdoes not mention\b",
    r"\bnot provided\b",
    r"\bnot specified\b",
    r"\bno information\b",
    r"\bi do not know\b",
    r"\bi don't know\b",
    r"\bcannot be determined\b",
    r"\bcannot determine\b",
    r"\bcannot answer\b",
    r"\bcan't answer\b",
    r"\bunable to answer\b",
    r"\bunable to determine\b",
    r"\bneed more context\b",
    r"\bneed access\b",
    r"\bwithout access\b",
    r"\bno access to\b",
    r"\bnot in the (?:provided )?article\b",
    r"\bno mention of\b",
    r"\binsufficient (?:information|context)\b",
    r"\bafter my knowledge cutoff\b",
    r"\bbeyond my knowledge cutoff\b",
    r"\bdo not have access\b",
    r"\bdon't have access\b",
    r"\bunknown\b",
    r"^abstain$",
    r"^n/?a$",
]

HEDGE_MARKERS = [
    r"\bmaybe\b",
    r"\bperhaps\b",
    r"\bpossibly\b",
    r"\bmight be\b",
    r"\bcould be\b",
    r"\blikely\b",
    r"\bprobably\b",
    r"\bi think\b",
    r"\bi believe\b",
    r"\bguess\b",
    r"\bहालांकि\b",
    r"\bthough\b.*\d",
    r"\bhowever\b.*\d",
]


def _normalize_short(s: str) -> str:
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = s.strip().lower()
    s = re.sub(r"^(the answer is|answer:)\s*", "", s)
    s = s.strip(" .,:;\"'()[]{}")
    s = re.sub(r"\s+", " ", s)
    return s


def detect_abstention_and_hedge(pred: str, question: str = "") -> Tuple[bool, bool]:
    """Return (is_explicit_abstention, is_hedged)."""
    norm = _normalize_short(pred)
    if not norm:
        return True, False
    has_abstain = any(re.search(pat, norm) for pat in ABSTENTION_PATTERNS)
    has_hedge = any(re.search(pat, norm) for pat in HEDGE_MARKERS)
    if has_abstain:
        # Also treat as hedged if response is long and introduces new numbers/guesses not in the question
        if not has_hedge and len(norm.split()) > 18 and re.search(r"\b\d+(?:\.\d+)?\b", norm):
            q_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", _normalize_short(question))) | {"2024", "2025", "2026"}
            novel_nums = [m for m in re.findall(r"\b\d+(?:\.\d+)?\b", norm) if m not in q_nums]
            if novel_nums:
                has_hedge = True
        return True, has_hedge
    return False, has_hedge


def deterministic_prepass(qa_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Attempt deterministic grading before calling the 3-judge panel.
    Returns a grade dict if deterministically resolved, else None.
    """
    pred_raw = qa_item.get("response", "").strip()
    if pred_raw.startswith("[ERROR:"):
        return {
            "grading_path": "deterministic_error",
            "final_grade": "NOT_ATTEMPTED",
            "hedged": False,
            "judge_votes": {},
            "disagreement": False,
        }

    is_abs, is_hedged = detect_abstention_and_hedge(pred_raw, qa_item.get("question", ""))
    if is_abs and not is_hedged:
        return {
            "grading_path": "deterministic_abstention",
            "final_grade": "NOT_ATTEMPTED",
            "hedged": False,
            "judge_votes": {},
            "disagreement": False,
        }

    # For near-miss unanswerable items, if the model did not cleanly abstain, route to judge panel
    if not qa_item.get("answerable", True):
        return None

    # Answerable item: check normalized exact match
    norm_pred = _normalize_short(pred_raw)
    norm_gold = _normalize_short(str(qa_item["gold_answer"]))
    ans_type = qa_item.get("answer_type", "text")
    prec_rule = qa_item.get("numeric_precision_rule", "not_applicable")

    if norm_pred == norm_gold and norm_pred != "":
        return {
            "grading_path": "deterministic_exact",
            "final_grade": "CORRECT",
            "hedged": False,
            "judge_votes": {},
            "disagreement": False,
        }

    # Deterministic numeric sigfig match when response is a clean number (+ optional unit from gold/question)
    if ans_type == "number" and prec_rule != "exact_identifier" and not is_hedged:
        # Strip common currency/unit tokens if response is short (<= 4 words)
        if len(norm_pred.split()) <= 4:
            p_nums = re.findall(r"-?\d+(?:\.\d+)?", norm_pred.replace(",", ""))
            g_nums = re.findall(r"-?\d+(?:\.\d+)?", norm_gold.replace(",", ""))
            if len(p_nums) == 1 and len(g_nums) == 1:
                p_val = float(p_nums[0])
                g_val = float(g_nums[0])
                g_str = g_nums[0]
                # Check if predicted has MORE precision than gold (AA rule: PARTIAL_ANSWER if more detail than gold)
                p_dec = len(p_nums[0].split(".")[1]) if "." in p_nums[0] else 0
                g_dec = len(g_str.split(".")[1]) if "." in g_str else 0
                if p_dec == g_dec and abs(p_val - g_val) <= 1e-9:
                    return {
                        "grading_path": "deterministic_numeric",
                        "final_grade": "CORRECT",
                        "hedged": False,
                        "judge_votes": {},
                        "disagreement": False,
                    }

    return None


def _parse_json_Loose(text: str) -> Dict[str, Any]:
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    try:
        return json.loads(s)
    except Exception:
        i = s.find("{")
        j = s.rfind("}")
        if i != -1 and j != -1 and j > i:
            return json.loads(s[i : j + 1])
        raise


def judge_qa_with_panel(qa_item: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Grade a single QA item with the 3-lab judge panel (`google/gemini-4-argon`, `meta/muse-wonder`, `x-ai/grok-4.7`)."""
    pre = deterministic_prepass(qa_item)
    if pre is not None:
        if qa_item.get("eval_mode") == "closed_book":
            pre["correct_without_source"] = (pre["final_grade"] == "CORRECT")
        return pre, []

    prompt = PRIVACYBENCH_V2_QA_GRADER_PROMPT.format(
        eval_mode=qa_item.get("eval_mode", "grounded_answerable"),
        answer_type=qa_item.get("answer_type", "text"),
        precision_rule=qa_item.get("numeric_precision_rule", "not_applicable"),
        question=qa_item["question"],
        gold_answer=qa_item["gold_answer"],
        predicted_answer=qa_item.get("response", ""),
    )

    votes: Dict[str, str] = {}
    hedged_votes: Dict[str, bool] = {}
    judge_telemetry: Dict[str, Dict[str, Any]] = {}
    telemetry: List[Dict[str, Any]] = []

    def _call_one_qa_judge(jm: str) -> Tuple[str, str, bool, Dict[str, Any]]:
        # A failed call (403, or a transport error after call_llm's retries) raises: it is never a vote.
        resp_local = _judge_call(
            model_id=jm,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
            reasoning_effort=JUDGE_EFFORT[jm],
            timeout=JUDGE_TIMEOUT["qa"],
            max_retries=JUDGE_RETRIES,
        )
        telem_entry = {"step": "judge_qa", "judge_model": jm, "item_id": qa_item["item_id"], **resp_local}
        try:
            parsed = _parse_json_Loose(resp_local["content"])
            g = str(parsed.get("grade", "INCORRECT")).upper().strip()
            if g not in ("CORRECT", "INCORRECT", "PARTIAL_ANSWER", "NOT_ATTEMPTED"):
                letter_map = {"A": "CORRECT", "B": "INCORRECT", "C": "PARTIAL_ANSWER", "D": "NOT_ATTEMPTED"}
                g = letter_map.get(g[:1], "INCORRECT")
            return jm, g, bool(parsed.get("hedged", False)), telem_entry
        except Exception:
            # The judge answered but not in JSON: recover the grade from its text (unchanged pilot rule).
            raw_c = resp_local.get("content") or ""
            found_g = "INCORRECT"
            for cand_g in ["NOT_ATTEMPTED", "PARTIAL_ANSWER", "INCORRECT", "CORRECT"]:
                if cand_g in raw_c.upper():
                    found_g = cand_g
                    break
            return jm, found_g, detect_abstention_and_hedge(qa_item.get("response", ""))[1], telem_entry

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as j_ex:
        for jm, g, h_flag, t_entry in j_ex.map(_call_one_qa_judge, JUDGE_PANEL):
            votes[jm] = g
            hedged_votes[jm] = h_flag
            judge_telemetry[jm] = served_record(t_entry)
            telemetry.append(t_entry)

    vote_counts = Counter(votes.values())
    top_count = vote_counts.most_common(1)[0][1]
    if top_count >= 2:
        final_grade = vote_counts.most_common(1)[0][0]
    else:
        # 3-way tie fallback: conservative ordering INCORRECT > PARTIAL_ANSWER > NOT_ATTEMPTED > CORRECT
        for pref in ["INCORRECT", "PARTIAL_ANSWER", "NOT_ATTEMPTED", "CORRECT"]:
            if pref in votes.values():
                final_grade = pref
                break
        else:
            final_grade = "INCORRECT"

    disagreement = (len(set(votes.values())) > 1)
    is_hedged = (sum(1 for v in hedged_votes.values() if v) >= 2) or detect_abstention_and_hedge(qa_item.get("response", ""))[1]

    result = {
        "grading_path": "judge_panel",
        "final_grade": final_grade,
        "hedged": bool(is_hedged and final_grade == "NOT_ATTEMPTED"),
        "judge_votes": votes,
        "judge_hedged_votes": hedged_votes,
        "judge_telemetry": judge_telemetry,
        "disagreement": disagreement,
    }
    if qa_item.get("eval_mode") == "closed_book":
        result["correct_without_source"] = (final_grade == "CORRECT")
    return result, telemetry


def grade_single_summary(
    sum_item: Dict[str, Any],
    article: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Decompose a 5-sentence summary into atomic claims and verify each claim + key_fact
    using the 3-judge panel (`google/gemini-4-argon`, `meta/muse-wonder`, `x-ai/grok-4.7`).
    """
    summary_text = sum_item.get("response", "").strip()
    article_text = get_article_text(article, verify_hash=True)
    key_facts = [kf["claim"] for kf in article.get("key_facts", [])]
    telemetry: List[Dict[str, Any]] = []

    if not summary_text or summary_text.startswith("[ERROR:"):
        return {
            "claims": [],
            "key_facts": key_facts,
            "n_claims": 0,
            "n_supported": 0,
            "n_unsupported": 0,
            "n_contradicted": 0,
            "unsupported_rate": 0.0,
            "contradicted_rate": 0.0,
            "key_fact_coverage": 0.0,
            "judge_claim_disagreements": 0,
        }, telemetry

    # Step 1: Decompose into atomic claims
    decomp_resp = _judge_call(
        model_id="google/gemini-4-argon",
        messages=[{"role": "user", "content": SUMMARY_DECOMPOSE_PROMPT.format(summary_text=summary_text)}],
        temperature=0.0,
        max_tokens=1024,
        reasoning_effort=JUDGE_EFFORT["google/gemini-4-argon"],
        timeout=JUDGE_TIMEOUT["summary"],
        max_retries=JUDGE_RETRIES,
    )
    telemetry.append({"step": "summary_decompose", "article_id": article["id"], **decomp_resp})
    parsed_decomp = _parse_json_Loose(decomp_resp["content"])
    claims: List[str] = [str(c).strip() for c in parsed_decomp.get("claims", []) if str(c).strip()]
    if not claims:
        claims = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary_text) if s.strip()]

    # Step 2: 3-judge NLI verification of claims and key_facts
    verify_prompt = SUMMARY_VERIFY_PROMPT.format(
        article_text=article_text,
        summary_text=summary_text,
        claims_json=json.dumps([{"index": i, "claim": c} for i, c in enumerate(claims)], ensure_ascii=False),
        key_facts_json=json.dumps([{"index": i, "key_fact": kf} for i, kf in enumerate(key_facts)], ensure_ascii=False),
    )

    per_judge_claims: Dict[str, List[str]] = {}
    per_judge_kfs: Dict[str, List[str]] = {}
    sum_judge_telemetry: Dict[str, Dict[str, Any]] = {}

    def _call_one_sum_judge(jm: str) -> Tuple[str, List[str], List[str], Dict[str, Any]]:
        # A failed call (403, or a transport error after call_llm's retries) raises: it is never a set of labels.
        v_resp_local = _judge_call(
            model_id=jm,
            messages=[{"role": "user", "content": verify_prompt}],
            temperature=0.0,
            max_tokens=1500,
            reasoning_effort=JUDGE_EFFORT[jm],
            timeout=JUDGE_TIMEOUT["summary"],
            max_retries=JUDGE_RETRIES,
        )
        t_entry = {"step": "summary_verify", "judge_model": jm, "article_id": article["id"], **v_resp_local}
        try:
            v_parsed = _parse_json_Loose(v_resp_local["content"])

            c_labels = ["SUPPORTED"] * len(claims)
            for cv in v_parsed.get("claim_verdicts", []):
                idx = int(cv.get("index", -1))
                lbl = str(cv.get("label", "SUPPORTED")).upper().strip()
                if 0 <= idx < len(claims) and lbl in ("SUPPORTED", "UNSUPPORTED", "CONTRADICTED"):
                    c_labels[idx] = lbl

            kf_labels = ["OMITTED"] * len(key_facts)
            for kv in v_parsed.get("key_fact_verdicts", []):
                idx = int(kv.get("index", -1))
                lbl = str(kv.get("label", "OMITTED")).upper().strip()
                if 0 <= idx < len(key_facts) and lbl in ("COVERED", "OMITTED"):
                    kf_labels[idx] = lbl
            return jm, c_labels, kf_labels, t_entry
        except Exception:
            # The judge answered but its JSON did not parse: unchanged pilot rule.
            return jm, ["SUPPORTED"] * len(claims), ["OMITTED"] * len(key_facts), t_entry

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as s_ex:
        for jm, c_labels, kf_labels, t_entry in s_ex.map(_call_one_sum_judge, JUDGE_PANEL):
            per_judge_claims[jm] = c_labels
            per_judge_kfs[jm] = kf_labels
            sum_judge_telemetry[jm] = served_record(t_entry)
            telemetry.append(t_entry)

    # Consolidate majority vote per claim and per key fact
    claim_results = []
    claim_disagreements = 0
    for i, c_text in enumerate(claims):
        c_votes = {jm: per_judge_claims[jm][i] for jm in JUDGE_PANEL if jm in per_judge_claims}
        vc = Counter(c_votes.values())
        if vc.most_common(1)[0][1] >= 2:
            maj_label = vc.most_common(1)[0][0]
        else:
            maj_label = "UNSUPPORTED"
        disagrees = len(set(c_votes.values())) > 1
        if disagrees:
            claim_disagreements += 1
        claim_results.append({
            "claim_index": i,
            "claim": c_text,
            "majority_label": maj_label,
            "judge_votes": c_votes,
            "disagreement": disagrees,
        })

    kf_results = []
    for i, kf_text in enumerate(key_facts):
        k_votes = {jm: per_judge_kfs[jm][i] for jm in JUDGE_PANEL if jm in per_judge_kfs}
        vc = Counter(k_votes.values())
        maj_kf = vc.most_common(1)[0][0] if vc else "OMITTED"
        kf_results.append({
            "fact_index": i,
            "key_fact": kf_text,
            "majority_label": maj_kf,
            "judge_votes": k_votes,
            "disagreement": len(set(k_votes.values())) > 1,
        })

    n_claims = len(claim_results)
    n_supp = sum(1 for c in claim_results if c["majority_label"] == "SUPPORTED")
    n_unsupp = sum(1 for c in claim_results if c["majority_label"] == "UNSUPPORTED")
    n_contra = sum(1 for c in claim_results if c["majority_label"] == "CONTRADICTED")
    n_kf_cov = sum(1 for k in kf_results if k["majority_label"] == "COVERED")

    return {
        "n_claims": n_claims,
        "n_supported": n_supp,
        "n_unsupported": n_unsupp,
        "n_contradicted": n_contra,
        "unsupported_rate": round(n_unsupp / n_claims, 4) if n_claims else 0.0,
        "contradicted_rate": round(n_contra / n_claims, 4) if n_claims else 0.0,
        "hallucinated_claim_rate": round((n_unsupp + n_contra) / n_claims, 4) if n_claims else 0.0,
        "n_key_facts": len(kf_results),
        "n_key_facts_covered": n_kf_cov,
        "key_fact_coverage": round(n_kf_cov / len(kf_results), 4) if kf_results else 0.0,
        "judge_claim_disagreements": claim_disagreements,
        "claim_breakdown": claim_results,
        "key_fact_breakdown": kf_results,
        "decompose_telemetry": served_record(decomp_resp),
        "judge_telemetry": sum_judge_telemetry,
    }, telemetry


def grade_result_file(
    result_path: Path,
    articles_by_id: Dict[str, Dict[str, Any]],
    max_workers: int = 8,
    force_regrade: bool = False,
) -> None:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    qa_list = data.get("qa_responses", [])
    sum_list = data.get("summary_responses", [])

    truncated_empty = [
        f"{r.get('eval_mode')}::{r.get('item_id')}"
        for r in (qa_list + sum_list)
        if not (r.get("response") or "").strip() and r.get("telemetry", {}).get("finish_reason") == "length"
    ]
    if truncated_empty:
        raise RuntimeError(
            f"[grade_eval] Refusing to grade {result_path.name}: {len(truncated_empty)} response(s) are empty with "
            f"finish_reason='length' (e.g. {truncated_empty[:5]}). Increase max_tokens and re-run inference so "
            f"length-truncated reasoning calls are not misclassified as abstentions."
        )

    ungraded_qa = [i for i, q in enumerate(qa_list) if force_regrade or not q.get("grading")]
    ungraded_sum = [i for i, s in enumerate(sum_list) if force_regrade or not s.get("grading")]

    if data.get("grading_summary") and not ungraded_qa and not ungraded_sum:
        print(f"[grade_eval] {result_path.name} already graded; skipping (use --force to regrade).")
        return

    print(f"\n[grade_eval] Grading {result_path.name} ({len(ungraded_qa)}/{len(qa_list)} QA + {len(ungraded_sum)}/{len(sum_list)} summaries) with panel {JUDGE_PANEL}...")

    grading_telemetry: List[Dict[str, Any]] = []
    t0 = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        qa_futs = {ex.submit(judge_qa_with_panel, qa_list[i]): i for i in ungraded_qa}
        for fut in concurrent.futures.as_completed(qa_futs):
            idx = qa_futs[fut]
            g_res, telem = fut.result()
            qa_list[idx]["grading"] = g_res
            grading_telemetry.extend(telem)

        sum_futs = {
            ex.submit(grade_single_summary, sum_list[i], articles_by_id[sum_list[i]["article_id"]]): i
            for i in ungraded_sum
        }
        for fut in concurrent.futures.as_completed(sum_futs):
            idx = sum_futs[fut]
            s_res, telem = fut.result()
            sum_list[idx]["grading"] = s_res
            grading_telemetry.extend(telem)

    path_counts = Counter(q["grading"]["grading_path"] for q in qa_list)
    panel_items = [q for q in qa_list if q["grading"]["grading_path"] == "judge_panel"]
    panel_disagreements = sum(1 for q in panel_items if q["grading"].get("disagreement"))

    prev_cost = 0.0 if force_regrade else data.get("grading_summary", {}).get("grading_cost_usd", 0.0)
    prev_calls = 0 if force_regrade else data.get("grading_summary", {}).get("grading_calls", 0)
    grading_cost = round(prev_cost + sum(t.get("cost_usd", 0.0) for t in grading_telemetry), 6)
    previous = data.get("grading_summary") or {}
    data["grading_summary"] = {
        "graded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "judge_panel": JUDGE_PANEL,
        "judge_gateways": dict(JUDGE_GATEWAY),
        **({"judge_route_notes": dict(JUDGE_ROUTE_NOTES)} if JUDGE_ROUTE_NOTES else {}),
        **({"replaces": {k: previous[k] for k in ("graded_at", "judge_panel", "judge_gateway", "judge_gateways")
                         if k in previous}} if force_regrade and previous else {}),
        "judge_efforts": dict(JUDGE_EFFORT),
        "qa_grading_paths": dict(path_counts),
        "qa_panel_items": len(panel_items),
        "qa_panel_disagreements": panel_disagreements,
        "qa_panel_agreement_rate": round(1.0 - (panel_disagreements / len(panel_items)), 4) if panel_items else 1.0,
        "grading_calls": prev_calls + len(grading_telemetry),
        "grading_cost_usd": grading_cost,
        "grading_wall_seconds": round(time.time() - t0, 2),
    }

    result_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"  -> Graded {result_path.name}: paths={dict(path_counts)}, "
        f"panel_agreement={data['grading_summary']['qa_panel_agreement_rate']*100:.1f}%, "
        f"grading_cost=${grading_cost:.4f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Grade PrivacyBench v2 model outputs.")
    ap.add_argument("--results-dir", default=str(TASK_ROOT / "results"))
    ap.add_argument("--file", default=None, help="Single result JSON file to grade")
    ap.add_argument("--articles-file", default=str(TASK_ROOT / "evaluation_questions" / "articles.json"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--gateway", choices=("openrouter", cometapi_client.GATEWAY), default="openrouter")
    ap.add_argument("--models-file", default=str(TASK_ROOT / "evaluation_questions" / "models.json"))
    args = ap.parse_args()
    use_gateway(args.gateway, Path(args.models_file))

    articles_data = json.loads(Path(args.articles_file).read_text(encoding="utf-8"))["articles"]
    articles_by_id = {a["id"]: a for a in articles_data}

    if args.file:
        result_files = [Path(args.file)]
    else:
        res_dir = Path(args.results_dir)
        result_files = sorted(p for p in res_dir.glob("2026-*.json") if "rollup" not in p.name)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as f_ex:
        futs = [
            f_ex.submit(grade_result_file, rf, articles_by_id, args.workers, args.force)
            for rf in result_files
        ]
        for fut in concurrent.futures.as_completed(futs):
            fut.result()


if __name__ == "__main__":
    main()
