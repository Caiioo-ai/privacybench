# HIPAA Limited Data Set — PII Redaction Compliance Benchmark

**No aspect of this repository constitutes legal advice.**

A 40-question wooden, deterministic compliance benchmark for HIPAA Limited
Data Set (§164.514(e)(2)) PII redaction filters. Tests both:

1. **Removal** of the 16 LDS identifier categories that MUST be stripped, and
2. **Preservation** of the LDS-retained carve-outs (city/state/ZIP, dates,
   ages, geographic units larger than street) that MUST NOT be redacted.

Unlike the PrivacyBench tasks in this repo (which use a gpt-4o judge to score
subjective "is this PII associated with an individual" reasoning), this
benchmark is **rule-based and deterministic**: every email is PHI, every phone
is PHI, every name is PHI. Designed so that a true HIPAA-LDS filter scores
≥99%.

---

## Why a separate benchmark from PrivacyBench

PrivacyBench tests **subjective PII judgment** — the model decides whether
something "qualifies as personal data." That works well for general consumer
PII filtering but is the wrong evaluation regime for HIPAA, which is
**definitional**:

- Every email IS PHI in a covered-entity workflow (no judgment call).
- Every name IS PHI (no judgment call).
- Every dose, every diagnosis date, every MRN — definitional.
- AND simultaneously: city/state/ZIP, full birth dates for non-elderly
  patients, ages, and geographic-coded regions larger than street are NOT
  removed under §164.514(e)(2). They are LDS carve-outs.

A general-PII filter that scores well on PrivacyBench will under-redact PHI
(missing wooden categories that PrivacyBench's subjective grader would also
miss) AND over-redact LDS carve-outs (because it has no concept of "LDS
preserves these"). Both are HIPAA non-conformances.

This benchmark makes both visible.

---

## Results — first 2026 run (deterministic scoring, 2026-04-28)

Three top-line metrics:

- **overall_score_pct** — arithmetic mean of (per-question score / 5 × 100).
- **hipaa_recall_pct** — % of expected PHI spans the model caught (any
  class). For HIPAA defensibility this must be 100%.
- **carveout_preservation_pct** — % of LDS retained signals that the model did
  NOT redact. Required for §164.514(e)(2) compliance. (privacy-filter has no
  carve-out logic and is not designed for HIPAA, so this column is N/A there.)

| Model (alphabetical) | params | overall | HIPAA recall | LDS carve-out preservation |
|---|---:|---:|---:|---:|
| caiioo-research/hipaa-lds-detector | 117M | **98.5** | **100.00%** | **100.00%** |
| openai/privacy-filter | 50M | 77.0 | 92.41% | n/a (no LDS-aware carve-out) |
| openai/privacy-filter + regex augment | 50M | 80.5 | 92.41% | n/a |

Notes:
- **HIPAA recall = 92.41%** for openai/privacy-filter on this benchmark means
  it misses ~7.6% of PHI spans the LDS standard requires removed. There is no
  acceptable miss rate for a covered-entity workflow under §164.514(e)(2).
  This is not a criticism of the privacy-filter model — it was not trained
  for HIPAA. The result is reported here so anyone considering using a
  general-PII filter in a HIPAA context can see the gap.
- **caiioo-research/hipaa-lds-detector** at 100% / 100% is the design point
  for an LDS-aware filter. The model is not published with this benchmark
  entry; the per-question result file in `results/` records what a model with
  these scores produced span-by-span and is reproducible from the questions
  in `evaluation_questions/`. The `caiioo-research/hipaa-lds-detector`
  identifier refers to a proprietary model owned by Six Cailloux, LLC. — see
  the repo-root `MODELS.md` for the full ownership / availability
  disclosure.

### Cross-task: the same LDS detector on canonical PrivacyBench

| Task | Score | What this means |
|---|---:|---|
| HIPAA LDS bench (this directory) | 98.5 | Designed for it. 100% recall, 100% carve-out preservation. |
| PrivacyBench 25-q (`../pii_redaction_task/`) | 68.8 | LDS preserves dates, city/state/ZIP, ages **by design**. PrivacyBench treats these as PII. The 68.8 is the LDS detector honoring HIPAA, not failing PrivacyBench. |

The 68.8 result is recorded in
`results/2026-04-28-caiioo-research-hipaa-lds-detector-on-privacybench.json`
specifically so readers can verify that what looks like a worse score is in
fact the LDS standard producing different outputs than PrivacyBench's
permissive "redact anything that could possibly identify someone" framing.

A model trained for one regulatory framework is expected to underperform
benchmarks based on a different framework. The same applies in reverse:
PrivacyBench-tuned models will overshoot LDS by removing carve-outs.

---

## Question set

40 questions, hand-authored. Each question has:

- `text` — the input string.
- `phi_spans` — list of `{start, end, label}` for items that MUST be removed.
- `retained_signals` — list of `{start, end, kind}` for LDS carve-outs that
  MUST NOT be removed.
- `category` — the test category (e.g. `name_basic`, `street_address_with_carveout`).

Categories cover all 16 LDS identifier types plus carve-out tests:

```
name_basic, name_compound, address (with/without carveout),
phone_or_fax, email, ssn_or_taxid, mrn, health_plan_id,
account_number, vehicle_id, device_id, ip, biometric_id,
dates_phi (DOB) vs dates_carveout (visit/discharge), age_carveout,
city_state_zip_carveout, geographic_carveout, etc.
```

See `evaluation_questions/hipaa_lds_bench.json` for the full schema and all 40
questions.

---

## Methodology

For each question, the predictor is given `text` and returns a list of spans
`{start, end, label}`. The scorer:

1. Matches each expected PHI span to a predicted span by **any character
   overlap** (boundary-forgiving — what matters is that the PHI is removed,
   not that the boundary is byte-perfect).
2. Counts: caught PHI, missed PHI, class-correct catches, retained-signal
   violations (predicted span overlapping a carve-out), spurious predictions
   (predicted span overlapping neither expected PHI nor a carve-out).
3. Applies the 0–5 rubric in `evaluation_questions/hipaa_lds_bench.json:
   description`:

   ```
   5  perfect: all PHI removed with right LDS class, all retained signals preserved
   4  all PHI removed but 1 class miscategorized OR 1 carve-out over-redaction
   3  all PHI removed but ≥2 issues
   2  ≥1 PHI missed (HIPAA violation)
   1  ≥2 PHI missed
   0  catastrophic
   ```

4. Aggregates `overall_score_pct`, `hipaa_recall_pct`,
   `carveout_preservation_pct` across all 40 questions.

There is no LLM grader. The benchmark is fully deterministic given the
predictor's outputs.

---

## Reproducibility

```bash
cd hipaa_lds_task/scripts
pip install -r ../../pii_redaction_heldout_2026/scripts/requirements.txt
# (this task reuses the redactor adapters from the heldout-2026 task)

python eval_hipaa_lds.py --target privacy-filter
python eval_hipaa_lds.py --target privacy-filter-regex
python eval_hipaa_lds.py --target hf-token-class:<your_repo_id>
```

The script is span-level (not text-redaction-level), so generative-model
targets via Ollama are not supported here — generative outputs don't preserve
the input string positions needed to compare against `phi_spans` /
`retained_signals`. Use the heldout-2026 task for those.

To add a new target with span-level outputs, write a `(text: str) -> list[Span]`
adapter in `../pii_redaction_heldout_2026/scripts/redactors_spans.py` and add
a dispatch case to `build_span_predictor`.

---

## Files

```
hipaa_lds_task/
├── README.md                            (this file)
├── evaluation_questions/
│   └── hipaa_lds_bench.json             40 q + ground-truth spans + carve-outs
├── scripts/
│   └── eval_hipaa_lds.py                deterministic scorer
└── results/                             per-target per-question rows + summary
    ├── 2026-04-28-caiioo-research-hipaa-lds-detector.json
    ├── 2026-04-28-openai-privacy-filter.json
    └── 2026-04-28-caiioo-research-hipaa-lds-detector-on-privacybench.json
```

---

## Citation

```
@misc{privacybench_hipaa_lds_2026,
  title = {HIPAA Limited Data Set — PII Redaction Compliance Benchmark},
  author = {{Caiioo-ai}},
  howpublished = {\url{https://github.com/Caiioo-ai/privacybench}},
  year = {2026},
}
```

Maintained under MIT (see `../LICENSE`). PrivacyBench is a trademark of
Alex J. Wall (see `../NOTICE`). Six Cailloux, LLC.
