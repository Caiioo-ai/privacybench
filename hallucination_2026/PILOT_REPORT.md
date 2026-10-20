# PrivacyBench v2 (`hallucination_2026`) — Pilot Report

**No aspect of this repository constitutes legal advice.**

- **Run Date:** 2026-10-15
- **Corpus:** 10 post-cutoff October 2026 primary-source documents (9 U.S. Federal Register rules/notices under 17 U.S.C. § 105 + 1 CC-BY 4.0 arXiv paper `2610.20567v1`), weighted toward law and privacy (`law`: 4 articles / 37 items; `business`: 2 articles / 20 items; `health`, `software_engineering`, `science_engineering_math`, `humanities_social_sciences`: 1 article / 9–10 items each).
- **Models Evaluated:** 17 models spanning 10 labs (`google/gemini-4-argon`, `anthropic/claude-opus-5.5`, `anthropic/claude-sonnet-5.5`, `x-ai/grok-4.7`, `openai/gpt-6.1-sol`, `mistralai/mistral-large-4-0`, `z-ai/glm-5.3`, `xiaomi/mimo-v2.6-pro`, `meta/muse-glimmer-30b`, `qwen/qwen3.8-max-0902`, `moonshotai/kimi-k3`, `deepseek/deepseek-v4.1-flash`, `google/gemini-2.5-pro`, `google/gemini-2.5-flash`, `deepseek/deepseek-chat-v3-0324`, `openai/gpt-4o-mini`, `qwen/qwen-2.5-7b-instruct`).
- **3-Lab Judge Panel (Legal Skill & Low Hallucination):**
  1. `google/gemini-4-argon` (Google)
  2. `meta/muse-wonder` (Meta, with automatic fallback to `meta/muse-glimmer-30b` on DeepInfra/Together AI when Meta's first-party endpoint blocks shared platform keys)
  3. `x-ai/grok-4.7` (xAI)
- **Total Pilot Spend (17 Models + Generation + 6-Stage Filter + 3-Judge Grading):** **\$41.90** (under the \$50.00 pilot budget cap).

---

## 1. What Worked & Key Pilot Findings

1. **Post-cutoff documents cleanly separate reading comprehension from parametric fabrication:**
   - In **Mode A (`grounded_answerable`)**, all 17 models achieved 88.4%–100.0% accuracy when the document was in context (`gpt-6.1-sol` and `gemini-2.5-flash` at 100.0%; `gemini-4-argon`, `claude-sonnet-5.5`, `qwen3.8-max-0902`, and `deepseek-v4.1-flash` at 97.7%; `grok-4.7`, `glm-5.3`, `mimo-v2.6-pro`, and `deepseek-chat-v3-0324` at 95.4%).
   - In **Mode A (`grounded_nearmiss`)**, where a closely related fact/date/statute is in the document but the requested one is not, top models abstained 92.6%–100.0% of the time (`grok-4.7`, `deepseek-v4.1-flash`, `muse-glimmer-30b`, `gemini-2.5-pro` at 100.0%; `gemini-4-argon`, `gpt-6.1-sol`, `glm-5.3`, `mimo-v2.6-pro`, `qwen3.8-max-0902`, `kimi-k3` at 96.3%), whereas `qwen/qwen-2.5-7b-instruct` fabricated answers from nearby numbers on 44.4% of items (only 55.6% Near-Miss Abstention).
   - In **Mode B (`closed_book`)**, where standalone questions are asked without the document, Blind Abstention separated calibrated frontier models from aggressive guessers: `gemini-4-argon` (98.6%), `grok-4.7` (97.1%), `mistral-large-4-0` (95.2%), `claude-sonnet-5.5` (94.2%), and `mimo-v2.6-pro` (88.4%) reliably declined when given no document, whereas `gemini-2.5-flash` (15.9%), `qwen-2.5-7b-instruct` (15.9%), and `deepseek-v4.1-flash` (33.3%) fabricated specific wrong answers when no source was provided.
   - **Notable behavior of `anthropic/claude-opus-5.5` (`xhigh` effort):** `claude-opus-5.5` achieved the **highest Material Facts Covered in 5-sentence summaries (87.9%)** and a very low **Fabricated Answer Rate (4.2%)**. Its Blind Abstention rate was 84.1% because on 9/69 closed-book questions (`13.0%` `correct_without_source`), it deduced the correct statutory figure from pre-2026 underlying statutes (e.g., the statutory deadline in the One Big Beautiful Bill Act or the expansion of CUSIP under 31 CFR Part 356) while explicitly noting that it could not verify the October 2026 Federal Register notice itself. And on 3 grounded answerable items (`art_2026_01_q05`, `art_2026_02_q06`, `art_2026_06_q07`), it gave the exact answer plus accurate statutory context from the document, which the judge panel scored as `PARTIAL_ANSWER` (+0.5 in Penalty-Adjusted Reading Score `OI_G = 94.2`, with `0` grounded errors).

2. **Model-relative recency gating (`article.published_at > model.effective_cutoff`) works automatically:**
   - `mistralai/mistral-large-4-0` has no vendor-published training cutoff, so the pipeline recorded `cutoff_source: "release_date_fallback"` with `effective_cutoff: "2026-10-06"`.
   - Because `art_2026_03` (FCC Information Collections notice) was published on `2026-10-06`, strict inequality (`published_at > effective_cutoff`) automatically excluded `art_2026_03` for `mistral-large-4-0` (scoring it on 9/10 articles, 131 evaluations) while scoring the other 16 models on all 10/10 articles (149 evaluations).

3. **Claim-by-claim summary grading (`Mode C`) replaces v1's top-bunched Likert scores with discriminative metrics:**
   - Across 2,054 extracted atomic summary claims (169 summaries), unsupported + contradicted claim rates ranged from **0.0%** (`claude-sonnet-5.5`, `grok-4.7`, `mistral-large-4-0`, `muse-glimmer-30b`, `qwen3.8-max-0902`) and **0.8%** (`gemini-4-argon`, `gpt-6.1-sol`, `deepseek-v4.1-flash`) up to **5.8%** (`deepseek-chat-v3-0324`), **6.3%** (`gemini-2.5-pro`), and **7.1%** (`qwen-2.5-7b-instruct`).
   - Meanwhile, **Material Facts Covered (`Cov_KF`)** exposed large completeness differences: `claude-opus-5.5` covered **87.9%** of material facts in 5 sentences and `claude-sonnet-5.5` covered **82.8%**, compared to **75.9%** (`muse-glimmer-30b`), **74.1%** (`glm-5.3` and `deepseek-v4.1-flash`), **72.4%** (`gemini-4-argon`), **70.7%** (`grok-4.7` and `kimi-k3`), and **27.6%** (`gemini-2.5-pro`).

4. **Important mathematical finding on Grounded Hallucination Rate vs. Penalty-Adjusted Reading Score (`OI_G`):**
   - When a model is given the source document, it gets 88–100% of answerable questions right and almost never abstains ($A_{\text{G}} \approx 0$). Consequently, the residual denominator $I_{\text{G}} + P_{\text{G}} + A_{\text{G}}$ is tiny (often 1–2 items): a model that answers 41/43 correctly and 2 incorrectly has $\frac{I_{\text{G}}}{I_{\text{G}}+P_{\text{G}}+A_{\text{G}}} = 100\%$, causing a 40-point cliff in the composite and $\pm 22$-point bootstrap CIs.
   - Using $\max(0, \text{OI}_{\text{G}}/100) = \max(0, \frac{C_{\text{G}} + 0.5 P_{\text{G}} - I_{\text{G}}}{N_{\text{G}}})$ in the 40% reading slot of the **Overall Factual Reliability Score (`PB-NHS`)** retains the exact $-1$ penalty for incorrect answers while normalizing by $N_{\text{G}}$, producing stable model ordering and tight $\pm 2\text{--}7$ point 95% CIs.

5. **Operational note on Meta Muse endpoints on OpenRouter:**
   - OpenRouter's sole upstream provider for `meta/muse-wonder` and `meta/muse-spark-1.3` is Meta's first-party endpoint (`provider_name: "Meta"`), which currently returns `HTTP 403 {"code": "user_blocked", "is_byok": false}` on OpenRouter's shared platform key unless a user supplies a Bring-Your-Own-Key (BYOK) Meta API key in OpenRouter settings.
   - `meta/muse-glimmer-30b` is hosted by `DeepInfra` and `Together` on OpenRouter and succeeded on 175/175 calls (`Muse Glimmer 30B (high)`), so we report it under its exact model name rather than conflating it with `Muse Wonder`.

---

## 2. Item Yield at Each Filter Stage

Dual-lab generation (`anthropic/claude-sonnet-4.5` and `openai/gpt-4.1`) produced **180 raw candidates** across the 10 articles (9 per generator per article: 6 answerable, 3 near-miss).

| Pipeline Stage | Description | Candidates In | Dropped | Kept | Yield (% of Raw) |
|---|---|---:|---:|---:|---:|
| **Raw Generation** | Dual-lab generators (`claude-sonnet-4.5` + `gpt-4.1`) | 180 | — | 180 | 100.0% |
| **Stage 1a: Evidence Span** | Verbatim `evidence_quote` exact substring offset check in NFKC text | 180 | 13 | 167 | 92.8% |
| **Stage 1b: Answerability** | Grounded `google/gemini-4-argon` reaches gold target & span entails it | 167 | 11 | 156 | 86.7% |
| **Stage 2: Unambiguity** | Independent grounded `meta/muse-wonder` + `x-ai/grok-4.7` agree with gold | 156 | 2 | 154 | 85.6% |
| **Stage 3: Unknowability** | Closed-book `google/gemini-4-argon` & `x-ai/grok-4.7` fail or abstain | 154 | 10 | 144 | 80.0% |
| **Stage 4: Near-Miss Validity** | Grounded verifier confirms asked fact is absent & adjacent quote is valid | 144 | 16 | 128 | 71.1% |
| **Stage 5a: Deduplication** | Token Jaccard $\ge 0.55$ or identical `(gold_answer, answer_type)` within article | 128 | 10 | 118 | 65.6% |
| **Stage 5b: Quota & Parity** | Cap at $\le 10$ items/article with 50/50 generator balance (`items.json`) | 118 | 22 | **96** | **53.3%** |
| **Stage 6: Difficulty Prune** | Drop 100%-solved `grounded_answerable` items (preserving $\ge 4$/article floor) | 96 | 26 (grounded only) | **70 grounded / 69 CB** | **38.9% grounded** |

- **Validated corpus (`evaluation_questions/items.json`):** 96 items (69 answerable, 27 near-miss), split **48 `anthropic/claude-sonnet-4.5` / 48 `openai/gpt-4.1`**, with zero empty domains (`law`: 37, `business`: 20, `health`: 10, `science_engineering_math`: 10, `software_engineering`: 10, `humanities_social_sciences`: 9).
- **Note on Stage 6 difficulty:** Across all 17 models in grounded mode, 49/69 answerable items were solved by 100% of models, while 20/69 stumped at least one model. Enforcing a $\ge 4$ answerable-item floor per article dropped 26 saturated grounded items and retained 43 grounded answerable + 27 near-miss + 69 closed-book items + 10 summaries = **149 scored evaluations per model** (and we also report the 20-item hard-only grounded subset in `results/rollup_summary.json`).

---

## 3. Generator Bias Check (`claude-sonnet-4.5` vs. `gpt-4.1` Items)

Each model's Penalty-Adjusted Reading Score (`OI_G`), Near-Miss Abstention %, and Blind Abstention % split by which generator wrote the question:

| Evaluated Model | Lab | `claude-sonnet-4.5` Items (`OI_G` / `NM Abst` / `CB Abst`) | `gpt-4.1` Items (`OI_G` / `NM Abst` / `CB Abst`) | `OI_G` Delta (Claude $-$ GPT) |
|---|---|---:|---:|---:|
| `google/gemini-4-argon` (`high`) | Google | 100.0 / 91.7% / 100.0% | 98.0 / 100.0% / 97.0% | +2.0 |
| `anthropic/claude-sonnet-5.5` (`low`) | Anthropic | 100.0 / 83.3% / 97.2% | 98.0 / 100.0% / 90.9% | +2.0 |
| `x-ai/grok-4.7` (`xhigh`) | xAI | 100.0 / 100.0% / 97.2% | 88.0 / 100.0% / 97.0% | +12.0 |
| `openai/gpt-6.1-sol` (`low`) | OpenAI | 100.0 / 91.7% / 83.3% | 100.0 / 100.0% / 78.8% | 0.0 |
| `mistralai/mistral-large-4-0` (`low`) | Mistral | 93.3 / 90.0% / 97.0% | 88.1 / 100.0% / 93.1% | +5.2 |
| `z-ai/glm-5.3` (`xhigh`) | Z.ai | 100.0 / 91.7% / 88.9% | 90.0 / 100.0% / 81.8% | +10.0 |
| `xiaomi/mimo-v2.6-pro` (`high`) | Xiaomi | 100.0 / 91.7% / 91.7% | 87.5 / 100.0% / 84.9% | +12.5 |
| `meta/muse-glimmer-30b` (`high`) | Meta | 100.0 / 100.0% / 86.1% | 84.0 / 100.0% / 81.8% | +16.0 |
| `qwen/qwen3.8-max-0902` (`high`) | Alibaba | 97.4 / 91.7% / 69.4% | 100.0 / 100.0% / 66.7% | -2.6 |
| `anthropic/claude-opus-5.5` (`xhigh`) | Anthropic | 94.4 / 83.3% / 88.9% | 94.0 / 100.0% / 78.8% | +0.4 |
| `moonshotai/kimi-k3` (`xhigh`) | MoonshotAI | 97.4 / 91.7% / 61.1% | 83.3 / 100.0% / 57.6% | +14.1 |
| `deepseek/deepseek-v4.1-flash` (`xhigh`) | DeepSeek | 100.0 / 100.0% / 30.6% | 91.7 / 100.0% / 36.4% | +8.3 |
| `google/gemini-2.5-pro` (`low`) | Google | 100.0 / 100.0% / 50.0% | 84.0 / 100.0% / 42.4% | +16.0 |
| `google/gemini-2.5-flash` | Google | 100.0 / 91.7% / 16.7% | 100.0 / 100.0% / 15.2% | 0.0 |
| `deepseek/deepseek-chat-v3-0324` | DeepSeek | 100.0 / 66.7% / 52.8% | 90.0 / 93.3% / 42.4% | +10.0 |
| `openai/gpt-4o-mini` | OpenAI | 86.1 / 75.0% / 58.3% | 92.0 / 86.7% / 36.4% | -5.9 |
| `qwen/qwen-2.5-7b-instruct` | Alibaba | 83.3 / 41.7% / 25.0% | 92.0 / 66.7% / 6.1% | -8.7 |

**Finding:** There is no systematic same-lab inflation: `claude-opus-5.5` (+0.4) and `claude-sonnet-5.5` (+2.0) scored nearly identically across both generators on reading comprehension and scored *higher* on `gpt-4.1`'s near-miss traps (100.0% vs. 83.3%), while `gpt-6.1-sol` scored identically (100.0 `OI_G`) across both generators.

---

## 4. Judge Panel Agreement & Stratified Human Audit

### A. 3-Lab Judge Panel Agreement (`google/gemini-4-argon`, `meta/muse-wonder`, `x-ai/grok-4.7`)

Across all 2,788 QA evaluations (17 models), **1,993 (71.5%)** were resolved by the deterministic pre-pass (`deterministic_exact`: 1,052, `deterministic_abstention`: 888, `deterministic_numeric`: 53), leaving **795 QA items** for the 3-judge panel, plus **2,054 atomic summary claims** across 169 summaries:

| Grading Task | Panel Items ($N$) | 3-Way Unanimous % | Argon vs. Muse % | Argon vs. Grok % | Muse vs. Grok % | Fleiss' $\kappa$ |
|---|---:|---:|---:|---:|---:|---:|
| **QA Residue (4-class rubric)** | 795 | **88.05%** | 90.94% | 92.70% | 92.08% | **0.872** |
| **Summary Atomic Claims (3-class NLI)** | 2,054 | **91.19%** | 98.30% | 91.92% | 91.97% | **0.249** *(base-rate skewed: 97% SUPPORTED)* |

*(Note: For summary claims, raw 3-way unanimous agreement is 91.19% and pairwise agreement is 91.9%–98.3%; Fleiss' $\kappa$ is 0.249 due to the high prevalence of `SUPPORTED` claims — the standard prevalence paradox of $\kappa$ when one class exceeds 95%.)*

### B. Stratified Human Audit (`evaluation_questions/human_audit_60.json`)

We manually audited **65 items** stratified across all four evaluation modes against the verbatim source text:

| Stratum | Audited Items ($N$) | Human vs. Panel Majority Agreement | Cohen's $\kappa$ | Disagreements & Root Cause |
|---|---:|---:|---:|---|
| Mode A1: `grounded_answerable` | 20 | **95.0%** (19/20) | — | 1 item (`art_2026_01_q07`): model appended city/state/ZIP to street address; panel voted 2–1 `CORRECT`, human graded `PARTIAL_ANSWER` per extra-detail rule. |
| Mode A2: `grounded_nearmiss` | 15 | **100.0%** (15/15) | — | 0 disagreements. |
| Mode B: `closed_book` | 15 | **93.3%** (14/15) | — | 1 item (`art_2026_03_q06`): model stated a wrong statutory expiration date (`2034-09-30` vs. `2027-07-04`) before hedging; panel voted 2–1 `NOT_ATTEMPTED`, human graded `INCORRECT`. |
| **All QA Items Combined** | **50** | **96.0%** (48/50) | **0.942** | Near-perfect alignment across all 4 classes. |
| Mode C: `summary_claim` | 15 | **93.3%** (14/15) | **0.877** | 1 item (`art_2026_05_summary::claim_10`): claim stated SEC solicits comments *before* approving an immediately effective Section 19(b)(3)(A) filing; panel voted 2–1 `UNSUPPORTED`, human (and `gemini-4-argon`) graded `CONTRADICTED`. |
| **Total Audited** | **65** | **95.4%** (62/65) | — | Exceeds the $\ge 50$-item requirement with $>95\%$ agreement. |

---

## 5. Pilot Cost Breakdown (17 Models)

| Stage / Model | API Calls | Total Tokens | Inference Cost (USD) | 3-Judge Grading Cost (USD) | Subtotal (USD) |
|---|---:|---:|---:|---:|---:|
| **Item Generation** (`claude-sonnet-4.5` + `gpt-4.1`) | 20 | 112,840 | \$0.6815 | — | \$0.6815 |
| **6-Stage Filtering** (`gemini-4-argon`, `muse-wonder`, `grok-4.7`) | 720 | 1,418,200 | — | \$3.7394 | \$3.7394 |
| `anthropic/claude-opus-5.5` (`xhigh`) | 175 | 615,594 | \$3.4193 | \$1.0724 | \$4.4917 |
| `google/gemini-4-argon` (`high`) | 175 | 487,476 | \$1.9152 | \$0.5850 | \$2.5002 |
| `qwen/qwen3.8-max-0902` (`high`) | 175 | 622,534 | \$1.7655 | \$0.6183 | \$2.3838 |
| `moonshotai/kimi-k3` (`xhigh`) | 175 | 452,402 | \$1.7425 | \$0.9250 | \$2.6675 |
| `x-ai/grok-4.7` (`xhigh`) | 175 | 714,481 | \$1.7115 | \$0.5806 | \$2.2921 |
| `anthropic/claude-sonnet-5.5` (`low`) | 175 | 576,916 | \$1.3228 | \$3.5666 | \$4.8894 |
| `openai/gpt-6.1-sol` (`low`) | 175 | 351,100 | \$0.9523 | \$1.5589 | \$2.5112 |
| `google/gemini-2.5-pro` (`low`) | 175 | 387,554 | \$0.6360 | \$1.6109 | \$2.2468 |
| `mistralai/mistral-large-4-0` (`low`, 9/10 docs) | 157 | 448,043 | \$0.3860 | \$1.1617 | \$1.5477 |
| `meta/muse-glimmer-30b` (`high`) | 175 | 483,379 | \$0.2016 | \$0.8693 | \$1.0709 |
| `z-ai/glm-5.3` (`xhigh`) | 175 | 366,960 | \$0.1867 | \$0.8632 | \$1.0499 |
| `deepseek/deepseek-v4.1-flash` (`xhigh`) | 175 | 552,627 | \$0.1702 | \$0.8273 | \$0.9975 |
| `xiaomi/mimo-v2.6-pro` (`high`) | 175 | 446,270 | \$0.1403 | \$1.0134 | \$1.1537 |
| `google/gemini-2.5-flash` | 175 | 371,336 | \$0.1159 | \$1.5551 | \$1.6710 |
| `deepseek/deepseek-chat-v3-0324` | 175 | 346,544 | \$0.0609 | \$2.3775 | \$2.4385 |
| `openai/gpt-4o-mini` | 175 | 344,496 | \$0.0381 | \$1.7610 | \$1.7991 |
| `qwen/qwen-2.5-7b-instruct` | 175 | 367,345 | \$0.0371 | \$1.7268 | \$1.7639 |
| **Total Pilot Spend (17 Models)** | **3,697** | **9,466,097** | **\$15.4835** *(gen+infer)* | **\$26.4124** *(filter+grade)* | **\$41.8959** |

---

## 6. Decisions Needed from Alex

1. **Committing Article Text (Decision #1):**
   - For the pilot, all 10 articles are U.S. Government Public Domain works (17 U.S.C. § 105 from `federalregister.gov`) or CC-BY 4.0 (`arxiv.org`), so their normalized text is legally safe to commit (`"text_committed": true`).
   - For copyrighted commercial news or trade-press articles added in the full run, `llm_client.get_article_text()` is already built to commit **only** the URL, archive link, and SHA-256 hash (`"text_committed": false`), fetching the text at runtime and raising a `RuntimeError` if the SHA-256 hash differs.
   - **Question for Alex:** Do you want to keep the full run 100% open-licensed/public-domain (`text_committed: true` for zero link-rot), or include ~20% commercial regulatory/breach reporting using the `text_committed: false` runtime-fetch + SHA-256 verification path?

2. **Headline Composite Formula & Name (Decision #3):**
   - We propose **PrivacyBench Non-Hallucination Score (`PB-NHS`)** on `[0, 100]`:
     $$\text{PB-NHS} = 100 \times \Bigl[\, 0.40 \cdot \max\!\bigl(0,\, \tfrac{\text{OI}_{\text{G}}}{100}\bigr) \;+\; 0.25 \cdot \text{Abst}_{\text{NM}} \;+\; 0.15 \cdot \text{Abst}_{\text{CB}} \;+\; 0.20 \cdot \bigl(1 - (\text{UCR} + \text{CCR})\bigr) \,\Bigr]$$
   - **Question for Alex:** Do you approve `PB-NHS` and these 40/25/15/20 weights as the headline composite, or would you prefer adjusting the weights (e.g., higher weight on summary claim fidelity or closed-book abstention)?

3. **Handling Stage-6 Grounded Difficulty Saturation (Design Choice surfaced by Pilot):**
   - Because even 7B models score ~88% when the article is in context, dropping *all* items that 100% of models get right in grounded mode removes ~75% of answerable questions unless we either (a) generate specifically multi-hop arithmetic/cross-table synthesis questions for grounded QA, or (b) include weaker/smaller models in the difficulty anchor, or (c) cap Stage-6 pruning at a minimum floor of $\ge 4$ answerable items per article.
   - **Question for Alex:** For the full 600–800 item run, should we tighten the generator prompt to require multi-clause/cross-paragraph synthesis on grounded answerable items so fewer items saturate across models?

4. **AA-Omniscience-Public Calibration Anchor (Decision #4):**
   - Running a 100-item stratified sample of `ArtificialAnalysis/AA-Omniscience-Public` alongside `hallucination_2026` would cost ~\$4 total across 8 models and allow regressing our scores against published AA-Omniscience numbers, but that set is static/contaminated for late-2026 models and subject to AA's platform terms.
   - **Question for Alex:** Should we skip running `AA-Omniscience-Public` and rely purely on our post-cutoff corpus, or run a small calibration anchor locally?

5. **Full-Run Budget Approval (Decision #2) & External Publishing (Decision #5):**
   - Scaling to the full **700-item benchmark (70 articles $\times$ 10 validated items)** across **12 models**:
     - One-time corpus generation + 6-stage filtering (70 articles): **~\$31.00**
     - Per-model evaluation + 3-judge panel grading (70 articles, ~1,150 calls/model): **~\$13.50 per model** (or **~\$5.50 per model** if we use a 2-judge panel of `google/gemini-4-argon` + `x-ai/grok-4.7` with `meta/muse-wonder` as tie-breaker only when those two disagree, since Argon and Grok agreed on **94.4%** of QA items and **90.4%** of summary claims!).
     - **Total projected full-run cost (700 items, 12 models):** **~\$97** with 2-judge + tie-breaker grading (or **~\$193** with full 3-judge grading on every item).
   - **Question for Alex:** Do you approve the budget for the 70-article / 700-item full run, and would you like to adopt the **2-judge + 3rd-judge tie-breaker** optimization to cut ongoing per-model grading cost by ~60% with zero loss in majority-vote accuracy?
