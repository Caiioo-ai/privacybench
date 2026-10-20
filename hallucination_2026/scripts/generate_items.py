"""
Dual-lab question and key-fact generator for PrivacyBench v2 (`hallucination_2026`).

Follows AA-Omniscience's automated question generation discipline, extended to:
1. Two generator models from distinct labs (`anthropic/claude-sonnet-4.5` and `openai/gpt-4.1`)
   so generator bias can be measured and audited.
2. Three evaluation modes per article:
   - Mode A1: Grounded Answerable QA (hard, unambiguous, short exact answer, buried/table/qualified facts)
   - Mode A2: Grounded Near-Miss Unanswerable QA (adjacent distractor fact is in the article, asked fact is not)
   - Mode B:  Closed-Book QA (uses the standalone-phrased Mode A1 questions without the article)
   - Mode C:  5-Sentence Summary key-fact extraction (validated atomic key facts for coverage scoring)
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llm_client import TASK_ROOT, call_llm, get_article_text, normalize_text

DEFAULT_GENERATORS = [
    "anthropic/claude-sonnet-4.5",
    "openai/gpt-4.1",
]

GENERATION_PROMPT = """You are an expert benchmark question author for PrivacyBench v2, a rigorous factual reliability and hallucination evaluation modeled on AA-Omniscience.

You are given a recent primary-source article from domain `{domain}` (subtopic: `{subtopic}`), published on `{published_at}` titled "{title}".

Your task is to generate:
1. **6 Hard Answerable Questions** (`answerable: true`) from this article.
2. **3 Near-Miss Unanswerable Questions** (`answerable: false`) from this article.
3. **5 Key Facts** (`key_facts`) representing the core factual takeaways of the article for evaluating a 5-sentence summary.

### Strict Rules for Answerable Questions (`answerable: true`)
- **Single unambiguous answer:** There must be exactly one indisputable factual answer supported by the article.
- **Self-contained for both Grounded and Closed-Book modes:** Include full context in the question itself (exact document/case/agency name, date, entity names, or version/matter numbers) so the question is well-defined even without the article attached.
- **Short exact answer in a stated format:** The question MUST explicitly state the required format or units (e.g., "answer as an integer", "in USD millions to one decimal place", "in YYYY-MM-DD format", "exact statutory/order section identifier", or "full organization/person name"). Gold answers must be 1 to 5 words/tokens.
- **No source-metadata trivia:** Do NOT ask about page numbers, author bylines, URLs, or paragraph counts. Exception: for `law` (and regulatory orders/statutes), specific section numbers, CFR citations, docket numbers, or article clauses DO count and are encouraged.
- **High difficulty:** Target facts buried in tables, late paragraphs, secondary mentions, figures with explicit qualifiers ("excluding X", "prior to offset", "for Tier 2 entities"), or numbers situated near plausible distractor numbers.
- **Verbatim evidence quote:** Provide an exact verbatim substring (`evidence_quote`) copied character-for-character from the article that entails the gold answer.

### Strict Rules for Near-Miss Unanswerable Questions (`answerable: false`)
- **Adjacent fact present, requested fact absent:** Anchor the question on a specific entity, table, metric, or regulatory requirement in the article where an *adjacent* fact IS stated, but the specific asked attribute is NOT stated anywhere in the article.
  - Examples: Q2 figure is given, ask for Q3; civil penalty for Company A is given, ask for the penalty imposed on Company B or by a different regulator; compliance deadline for large entities is given, ask for the exemption threshold for micro-entities; hardware latency on CPU is given, ask for GPU energy consumption in watts.
- **Do NOT write absurd questions:** The question must sound completely plausible and specific to the document so an uncalibrated model is tempted to extrapolate or guess from the adjacent number/fact.
- **Gold answer:** Must be `"not stated"`.
- **Verbatim adjacent evidence quote:** `evidence_quote` MUST be the exact verbatim substring from the article containing the *adjacent distractor fact*.
- **`near_miss_target_asked`:** Briefly state what specific unstated detail the question asks for and why it is absent.

### Output JSON Format
Return ONLY a valid JSON object matching this exact schema (no markdown fences, no commentary):
{{
  "answerable_questions": [
    {{
      "question": "...",
      "gold_answer": "...",
      "answer_type": "number | date | name | identifier | text",
      "numeric_precision_rule": "exact_integer | last_sigfig | exact_identifier | not_applicable",
      "evidence_quote": "exact verbatim substring from article"
    }}
  ],
  "nearmiss_questions": [
    {{
      "question": "...",
      "gold_answer": "not stated",
      "answer_type": "text",
      "numeric_precision_rule": "not_applicable",
      "evidence_quote": "exact verbatim substring of the adjacent distractor fact in the article",
      "near_miss_target_asked": "description of the asked attribute that is absent from the article"
    }}
  ],
  "key_facts": [
    {{
      "claim": "Concise, atomic statement of a key fact from the article.",
      "evidence_quote": "exact verbatim substring from article"
    }}
  ]
}}

Article Text:
{article_text}
"""


def find_quote_offsets(norm_article: str, quote: str) -> Tuple[Optional[int], Optional[int], str]:
    """
    Locate `quote` inside `norm_article`.
    Returns (start, end, matched_verbatim_substring) or (None, None, quote).
    """
    q_norm = normalize_text(quote)
    if not q_norm:
        return None, None, quote

    idx = norm_article.find(q_norm)
    if idx != -1:
        return idx, idx + len(q_norm), norm_article[idx : idx + len(q_norm)]

    # Case-insensitive fallback
    idx_ci = norm_article.lower().find(q_norm.lower())
    if idx_ci != -1:
        return idx_ci, idx_ci + len(q_norm), norm_article[idx_ci : idx_ci + len(q_norm)]

    # Whitespace-flexible regex fallback
    tokens = [re.escape(tok) for tok in q_norm.split() if tok]
    if len(tokens) >= 4:
        pattern = r"\s+".join(tokens)
        m = re.search(pattern, norm_article, flags=re.IGNORECASE)
        if m:
            return m.start(), m.end(), norm_article[m.start() : m.end()]

        # Try first 8 and last 6 tokens if quote had minor ellipsis
        if len(tokens) >= 12:
            head_pat = r"\s+".join(tokens[:6])
            tail_pat = r"\s+".join(tokens[-5:])
            mh = re.search(head_pat, norm_article, flags=re.IGNORECASE)
            if mh:
                mt = re.search(tail_pat, norm_article[mh.start() : mh.start() + 1200], flags=re.IGNORECASE)
                if mt:
                    s = mh.start()
                    e = mh.start() + mt.end()
                    return s, e, norm_article[s:e]

    return None, None, q_norm


def parse_json_from_llm(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(text[s : e + 1])
        raise


def generate_for_article(
    article: Dict[str, Any],
    generators: List[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Run each generator model on `article`, returning:
    (candidate_items, consolidated_key_facts, call_telemetry_list)
    """
    norm_text = get_article_text(article, verify_hash=True)
    art_id = article["id"]
    domain = article["domain"]
    subtopic = article["subtopic"]

    candidates: List[Dict[str, Any]] = []
    all_key_facts: List[Dict[str, Any]] = []
    telemetry_log: List[Dict[str, Any]] = []

    q_counter = 1
    for gen_model in generators:
        prompt = GENERATION_PROMPT.format(
            domain=domain,
            subtopic=subtopic,
            published_at=article["published_at"],
            title=article["title"],
            article_text=norm_text,
        )
        print(f"  [generate] article={art_id} generator={gen_model}")
        resp = call_llm(
            model_id=gen_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=3000,
            timeout=120,
        )
        telemetry_log.append({
            "step": "generate",
            "article_id": art_id,
            "generator_model": gen_model,
            **{k: v for k, v in resp.items() if k != "content"},
        })

        parsed = parse_json_from_llm(resp["content"])

        for aq in parsed.get("answerable_questions", []):
            s_off, e_off, matched_quote = find_quote_offsets(norm_text, aq.get("evidence_quote", ""))
            candidates.append({
                "id": f"{art_id}_cand_{q_counter:02d}",
                "article_id": art_id,
                "domain": domain,
                "subtopic": subtopic,
                "modes": ["grounded_answerable", "closed_book"],
                "question": aq["question"].strip(),
                "gold_answer": str(aq["gold_answer"]).strip(),
                "answer_type": aq.get("answer_type", "text"),
                "numeric_precision_rule": aq.get("numeric_precision_rule", "not_applicable"),
                "evidence_quote": matched_quote,
                "evidence_offsets": [s_off, e_off] if s_off is not None else None,
                "answerable": True,
                "near_miss_target_asked": None,
                "generator_model": gen_model,
                "validator_verdicts": {},
            })
            q_counter += 1

        for nm in parsed.get("nearmiss_questions", []):
            s_off, e_off, matched_quote = find_quote_offsets(norm_text, nm.get("evidence_quote", ""))
            candidates.append({
                "id": f"{art_id}_cand_{q_counter:02d}",
                "article_id": art_id,
                "domain": domain,
                "subtopic": subtopic,
                "modes": ["grounded_nearmiss"],
                "question": nm["question"].strip(),
                "gold_answer": "not stated",
                "answer_type": "text",
                "numeric_precision_rule": "not_applicable",
                "evidence_quote": matched_quote,
                "evidence_offsets": [s_off, e_off] if s_off is not None else None,
                "answerable": False,
                "near_miss_target_asked": nm.get("near_miss_target_asked", ""),
                "generator_model": gen_model,
                "validator_verdicts": {},
            })
            q_counter += 1

        for kf in parsed.get("key_facts", []):
            s_off, e_off, matched_quote = find_quote_offsets(norm_text, kf.get("evidence_quote", ""))
            if s_off is not None and len(all_key_facts) < 6:
                # Avoid duplicate key facts across generators
                claim_str = kf.get("claim", "").strip()
                if claim_str and not any(_token_jaccard(claim_str, existing["claim"]) > 0.45 for existing in all_key_facts):
                    all_key_facts.append({
                        "fact_id": f"{art_id}_kf{len(all_key_facts) + 1:02d}",
                        "claim": claim_str,
                        "evidence_quote": matched_quote,
                        "evidence_offsets": [s_off, e_off],
                        "generator_model": gen_model,
                    })

    return candidates, all_key_facts, telemetry_log


def _token_jaccard(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-z0-9]+", a.lower()))
    tb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate candidate PrivacyBench v2 items using two lab models.")
    ap.add_argument(
        "--articles",
        default=str(TASK_ROOT / "evaluation_questions" / "articles.json"),
        help="Path to articles.json",
    )
    ap.add_argument(
        "--out",
        default=str(TASK_ROOT / "evaluation_questions" / "candidates_raw.json"),
        help="Output path for raw candidates JSON",
    )
    ap.add_argument(
        "--generators",
        nargs="+",
        default=DEFAULT_GENERATORS,
        help="Generator model IDs from distinct labs",
    )
    args = ap.parse_args()

    articles_path = Path(args.articles)
    articles_data = json.loads(articles_path.read_text(encoding="utf-8"))
    articles = articles_data["articles"]

    import concurrent.futures

    all_candidates: List[Dict[str, Any]] = []
    all_telemetry: List[Dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        fut_to_idx = {ex.submit(generate_for_article, art, args.generators): i for i, art in enumerate(articles)}
        results_by_idx: Dict[int, Any] = {}
        for fut in concurrent.futures.as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            results_by_idx[idx] = fut.result()

    for i, art in enumerate(articles):
        cands, kfs, telem = results_by_idx[i]
        all_candidates.extend(cands)
        all_telemetry.extend(telem)
        if not art.get("key_facts"):
            art["key_facts"] = kfs

    # Persist updated articles (with key_facts) and candidates_raw.json
    articles_path.write_text(json.dumps(articles_data, indent=2, ensure_ascii=False), encoding="utf-8")
    out_path = Path(args.out)
    total_cost = round(sum(t.get("cost_usd", 0.0) for t in all_telemetry), 4)
    out_path.write_text(
        json.dumps(
            {
                "_meta": {
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "generators": args.generators,
                    "total_candidates": len(all_candidates),
                    "generation_cost_usd": total_cost,
                },
                "candidates": all_candidates,
                "telemetry": all_telemetry,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n[generate] Wrote {len(all_candidates)} raw candidates to {out_path} (cost=${total_cost:.4f})")


if __name__ == "__main__":
    main()
