# Progress Notes — `hallucination_2026` (PrivacyBench v2)

## Session Log — 2026-10-15

### 1. Context & Orientation
- Read `README.md`, `MODELS.md`, `NOTICE`, `requirements.txt`.
- Read `bias_study/BIAS_STUDY_OVERVIEW.md` and `bias_study/bias_study.py`.
- Read `pii_redaction_heldout_2026/README.md`, `scripts/eval_heldout.py`, `scripts/redactors.py`, `scripts/redactors_spans.py`, `evaluation_questions/generalization_bench.json`, `results/2026-05-02-google-gemma4-e2b.json`.
- Read `hipaa_lds_task/README.md` and `hipaa_lds_task/scripts/eval_hipaa_lds.py`.
- Read `ArtificialAnalysis/AA-Omniscience-Public` dataset card (`README.md`, Apache-2.0) and the AA-Omniscience paper (`arXiv:2511.13029v1`, CC BY 4.0).

### 2. Judge Panel Selection (Updated per Alex's Guidance)
- Prioritizing legal skill and low hallucination across 3 distinct labs:
  1. `google/gemini-4-argon` (Google: Gemini 4 Argon)
  2. `meta/muse-wonder` (Meta: Muse Wonder)
  3. `x-ai/grok-4.7` (xAI: Grok 4.7)
- Used both in the 6-stage question validation pipeline (`filter_items.py`) and in the post-prepass QA residue + summary claim NLI judge panel (`grade_eval.py`).

### 3. Pilot Corpus (`evaluation_questions/articles.json`)
- 10 primary-source documents published in October 2026 (`2026-10-06` to `2026-10-15`), all public domain (17 U.S.C. § 105 from `federalregister.gov`) or CC-BY 4.0 (`arxiv.org`):
  - `art_2026_01` (`law`, `2026-10-15`): DOJ Privacy Act of 1974; Systems of Records (`2026-21055`)
  - `art_2026_02` (`law`, `2026-10-07`): Treasury/IRS Privacy Act of 1974; System of Records (`2026-20494`)
  - `art_2026_03` (`law`, `2026-10-06`): FCC Information Collections Under Review (`2026-20486`)
  - `art_2026_04` (`law`, `2026-10-15`): OGE Post-Employment Conflict of Interest Restrictions (`2026-21059`)
  - `art_2026_05` (`business`, `2026-10-14`): SEC Texas Stock Exchange LLC Rule Filing (`2026-20923`)
  - `art_2026_06` (`business`, `2026-10-15`): Treasury Marketable Book-Entry Bills, Notes, and Bonds (`2026-21069`)
  - `art_2026_07` (`health`, `2026-10-15`): HHS/FDA Emergency Use Authorization for Four Animal Drugs (`2026-21084`)
  - `art_2026_08` (`software_engineering`, `2026-10-14`): arXiv:2610.20567v1 *Prioritizing Microservice Requests via Retries* (CC-BY 4.0)
  - `art_2026_09` (`science_engineering_math`, `2026-10-14`): FAA Airworthiness Directives; Bombardier Airplanes (`2026-21010`)
  - `art_2026_10` (`humanities_social_sciences`, `2026-10-15`): OPM Family and Medical Leave Act Final Rule (`2026-21104`)

### 4. Pilot Models (`evaluation_questions/models.json`)
- 8 models spanning 6 labs and 3 size classes (Frontier, Mid-Size, Small Open-Weight):
  1. `anthropic/claude-sonnet-5.5` (Frontier, vendor cutoff `2026-04-01`, 10/10 articles eligible)
  2. `openai/gpt-6.1-sol` (Frontier, vendor cutoff `2026-03-01`, 10/10 articles eligible)
  3. `google/gemini-2.5-pro` (Frontier, vendor cutoff `2025-01-01`, 10/10 articles eligible)
  4. `mistralai/mistral-large-4-0` (Frontier, release fallback `2026-10-06`, 9/10 articles eligible — `art_2026_03` published `2026-10-06` is filtered out by strict `published_at > effective_cutoff`)
  5. `deepseek/deepseek-chat-v3-0324` (Open-Weight Large, release fallback `2025-03-24`, 10/10 articles eligible)
  6. `google/gemini-2.5-flash` (Mid-Size, vendor cutoff `2025-01-01`, 10/10 articles eligible)
  7. `openai/gpt-4o-mini` (Mid-Size, vendor cutoff `2023-10-01`, 10/10 articles eligible)
  8. `qwen/qwen-2.5-7b-instruct` (Small Open-Weight 7B, release fallback `2024-10-16`, 10/10 articles eligible)

### 5. Item Generation & 6-Stage Filtering Results
- **Generation (`generate_items.py`):**
  - Dual-lab generators: `anthropic/claude-sonnet-4.5` (90 raw candidates) and `openai/gpt-4.1` (90 raw candidates) = **180 raw candidates** (120 answerable, 60 near-miss unanswerable), plus 5 validated `key_facts` per article.
  - Generation cost: **\$0.6815**.
- **Filtering (`filter_items.py` using `google/gemini-4-argon`, `meta/muse-wonder`, `x-ai/grok-4.7`):**
  - `raw_candidates`: 180 (`raw_answerable`: 120, `raw_nearmiss`: 60)
  - `dropped_stage1_evidence_span`: 13 (exact substring offset check in NFKC-normalized text)
  - `dropped_stage1_answerability`: 11 (grounded `google/gemini-4-argon` did not reach gold answer or span did not entail it)
  - `dropped_stage2_unambiguity`: 2 (independent grounded `meta/muse-wonder` and `x-ai/grok-4.7` disagreed with gold answer)
  - `dropped_stage3_unknowability`: 10 (closed-book `google/gemini-4-argon` or `x-ai/grok-4.7` guessed/deduced the answer without the article)
  - `dropped_stage4_nearmiss_validity`: 16 (near-miss fact was partially inferable or adjacent quote did not match)
  - `dropped_stage5_deduplication`: 10 (normalized token Jaccard $\ge 0.55$ or identical gold target + answer type within an article)
  - `dropped_stage5_quota_balance`: 22 (balanced per-article cap of $\le 10$ items with generator parity)
  - **Kept validated items (`evaluation_questions/items.json`):** **96 items** (69 answerable, 27 near-miss; exact 48/48 split between `anthropic/claude-sonnet-4.5` and `openai/gpt-4.1`; all 6 domains represented: `law` 37, `business` 20, `health` 10, `science_engineering_math` 10, `software_engineering` 10, `humanities_social_sciences` 9).
  - Filtering cost: **\$3.7394**.

### 6. Pilot Execution, Grading & Human Audit Summary
- **Inference (`run_eval.py`):** 8 models evaluated across eligible post-cutoff articles (`mistral-large-4-0`: 9/10 articles eligible; other 7 models: 10/10 articles eligible). Total inference cost: **\$3.5491**.
- **Grading (`grade_eval.py`):** Deterministic pre-pass resolved 67.2% of QA items; 3-lab judge panel (`google/gemini-4-argon`, `meta/muse-wonder` with `meta/muse-glimmer-30b` Together AI fallback on 403, `x-ai/grok-4.7`) achieved **90.87% 3-way unanimous agreement** ($\kappa = 0.895$) on QA residue and **89.65% 3-way unanimous agreement** on 937 atomic summary claims. Total grading cost: **\$15.3185**.
- **Human Audit (`human_audit.py` / `human_audit_60.json`):** 65 stratified items manually audited across all 4 modes; **95.4% (62/65)** agreement with 3-judge panel majority vote (**96.0%**, $\kappa = 0.942$ on QA items; **93.3%**, $\kappa = 0.877$ on summary claims).
- **Total Pilot Spend:** **\$23.2885** (under \$50 budget cap).

### 7. Deliverable Checklist
- [x] Read existing tasks and AA-Omniscience paper/dataset card
- [x] Deliverable 1: `hallucination_2026/README.md` + `NOTICE` attribution
- [x] Deliverable 2: Scripts in `hallucination_2026/scripts/` (`llm_client.py`, `generate_items.py`, `filter_items.py`, `run_eval.py`, `grade_eval.py`, `rollup_and_plot.py`, `human_audit.py`, `requirements.txt`)
- [x] Deliverable 3: Pilot execution (`generate_items.py`, `filter_items.py`, `run_eval.py`, `grade_eval.py`, `human_audit.py`, `rollup_and_plot.py`)
- [x] Deliverable 4: Short report (`PILOT_REPORT.md` + final summary)
