# PII Redaction — Held-Out Generalization Suite (2026)

**No aspect of this repository constitutes legal advice.**

A 150-question held-out evaluation of personal-data redaction across five axes
that the original 25-question PrivacyBench (`../pii_redaction_task/`) does not
specifically stress: dense formal documents, multilingual chat, code with
embedded secrets, and negative-control prose where any redaction is a false
positive.

This task complements the canonical 25-question PrivacyBench. Questions are
hand-authored, never used in training of any model whose results appear here,
and graded with the same gpt-4o judge methodology PrivacyBench uses.

---

## Question set

| Bucket | Questions | What it stresses |
|---|---:|---|
| `generalization_bench.json` | 30 | Mixed chat/document/multilingual, the catch-all OOD set |
| `doc_bench.json` | 30 | Dense formal documents (memos, letters, contracts, medical notes, legal filings, reports) |
| `multilingual_bench.json` | 40 | 10 languages × 4 contexts (es / de / fr / ja / zh / ko / ar / pt / ru / hi) |
| `code_bench.json` | 20 | API keys / DB strings / secrets files / SQL with PII |
| `negative_bench.json` | 30 | Public figures, fiction, form templates, corporate boilerplate, abstract prose ABOUT PII |
| **Total** | **150** | |

Authored 2026-04-30. Each question carries an `expected_pii` ground-truth list
(empty for negative controls). The grader scores against that list, not against
a free-text "correct answer" — this lets us measure both recall (did the model
catch the listed items) and false-positive control (did it leave the rest alone)
in a single 0–5 score.

### Synthetic-secret convention in `code_bench.json`

The code-paste questions test whether redactors recognize API-key-shaped
strings as auth secrets. To avoid tripping git-host secret scanners (which
match literal vendor prefixes like `sk_live_` and `xoxb-`), the bench uses
prefix-mutated synthetic equivalents — `sk_demo_…`, `xoxz-…`, `pk_demo_…`,
`AIdemo…`, `ghd_…` — that preserve the structural shape (long alphanumeric
suffix, recognizable prefix family) but do not match the exact regexes a
secret scanner uses for live keys. A correctly-trained redactor should still
classify them as auth secrets; the bench grades on whether the model removed
them, not on whether the literal vendor prefix is present.

---

## Results — first 2026 run (gpt-4o judge, 2026-05-02)

Per-bucket scores 0–100 (higher is better). Mean is the arithmetic mean of
the five bucket scores. Per-question rows for every entry are in `results/`.

| Model (alphabetical) | params | gen | doc | multi | code | negative | **mean** |
|---|---:|---:|---:|---:|---:|---:|---:|
| caiioo-research/personal-data-detector-v14 (PT) | 117M | 82.67 | 82.67 | 88.50 | 84.00 | 100.00 | **87.57** |
| caiioo-research/personal-data-detector-v14 (int8 ONNX) | 117M | 81.33 | 80.67 | 84.50 | 83.00 | 100.00 | **85.90** |
| google/gemma4:e2b (generative, via Ollama) | ~5B | 71.33 | 88.67 | 85.00 | 87.00 | 40.00 | **74.40** |
| openai/privacy-filter (token-class) | 50M | 74.67 | 74.00 | 67.50 | 69.00 | 93.33 | **75.70** |
| qwen3.5:0.8b (generative, via Ollama) | 0.8B | 18.00 | 16.00 | 16.00 | 29.00 | 66.67 | **29.13** |

Notes:
- **PT vs int8 ONNX** — the same model evaluated through PyTorch (full
  precision, training-time path) vs through int8-quantized ONNX (deployment
  path). The delta is the cost of quantization for this architecture.
- **gemma4:e2b** is generative redaction (free-text completion). Its low
  negative-bench score reflects that the negative-control questions ask the
  model to NOT redact (public figures in news, fictional addresses, corporate
  boilerplate). Generative redactors with a strong "redact aggressively" prior
  have systematically lower negative-bench scores.
- **qwen3.5:0.8b** is generative redaction at small scale; the bench is harder
  than its parameter budget supports. Included for the lower bound it
  establishes for tiny generative models on this task.
- The **canonical 25-question PrivacyBench** lives at
  `../pii_redaction_task/`. Same-model scores there can be compared against
  the held-out scores here to see how each model handles broader distribution
  shift.

---

## Methodology

1. **Inference** — for each question, run the target redactor and capture its
   redacted output (text-out, text-in).

2. **Grading** — pass `(original, expected_pii, redacted)` to the gpt-4o judge
   with a fixed prompt (see `scripts/eval_heldout.py:grade_redaction`). The
   judge returns `Grade: <0–5>` with a one-sentence justification.

3. **Scoring** — `(grade / 5.0) * 100` per question; arithmetic mean per bucket;
   arithmetic mean of bucket overalls for the suite mean.

4. **Output** — `results/held_out_suite_<label>.json` records per-question rows
   (id, context, expected_count, score_pct, judge feedback, first 300 chars of
   the redacted output, inference time) plus per-bucket and overall scores.

The grading prompt and the 0–5 → percentage conversion match the convention
established by PrivacyBench's smaller-model leaderboard — see
`../pii_redaction_task/README-redaction.md`.

---

## Reproducibility

```bash
cd pii_redaction_heldout_2026/scripts
pip install -r requirements.txt          # see ../requirements.txt
echo "OPENAI_API_KEY=..." > ../.env       # or OPENROUTER_API_KEY for the judge

python eval_heldout.py --target privacy-filter
python eval_heldout.py --target privacy-filter-regex
python eval_heldout.py --target ollama:gemma4:e2b
python eval_heldout.py --target hf-token-class:<your_repo_id>
```

Runs cost approximately 150 grader API calls per target; the gpt-4o judge is
the dominant expense. Inference time depends on the target — `privacy-filter`
runs in ~30s on CPU; `ollama:gemma4:e2b` takes ~2-3 minutes on Apple Silicon.

To add a new target, write a `(text: str) -> str` adapter in
`scripts/redactors.py` and add a dispatch case to `build_redactor`.

---

## Held-out hygiene — methodology note

A held-out evaluation set is only meaningful if the model under test was not
trained on it (or on examples shaped like it). When a model owner authors both
the training data and the held-out questions, surface-form contamination is
easy to introduce accidentally — the result looks like generalization but is
partly memorization.

While preparing one of the entries in this directory, the model owner observed
contamination between recently added training batches and the held-out
questions. The mitigation pattern is reusable by anyone in this situation:

**5–7-word shingle audit.** Build the union of all 5-, 6-, and 7-token
whitespace-tokenized shingles in the training corpus. For each held-out
question, count its overlapping shingles with that set. A threshold of ≥3
shared 5–7-grams is a sensitive indicator of structural overlap. Any held-out
question above that threshold should either (a) be rewritten in the held-out
set, or (b) the training data structurally orthogonalized so the overlap
drops below it. The same audit can detect inadvertent cross-contamination
between siblings: an OOD bench should not share verbatim phrasing with the
training set, with another bench in the same suite, or with publicly indexable
sources the model was pretrained on.

In the case observed, contamination was found in 23 of the 150 questions
(typical offenders: identical document templates such as "GDPR Data Subject
Request — Response Letter", verbatim corporate-address examples, and shared
fictional-character framing). The training batches were rewritten with
structurally-orthogonal alternatives until the audit produced zero hits at the
≥3-shingle threshold, the model was retrained, and the held-out scores were
re-measured. The decline on the contaminated subset between the
contamination-prone version and the cleaned version (typically 30–60 points
per question) is a calibration signal for how much memorization the original
score reflected.

We don't claim the audit is sufficient — semantic-level contamination
(paraphrased questions, new entities in the same template) can pass a
shingle-overlap test. But it catches the easy contamination that, if missed,
inflates results enough to invalidate comparisons.

---

## Files

```
pii_redaction_heldout_2026/
├── README.md                       (this file)
├── evaluation_questions/
│   ├── generalization_bench.json   30 q
│   ├── doc_bench.json              30 q
│   ├── multilingual_bench.json     40 q
│   ├── code_bench.json             20 q
│   └── negative_bench.json         30 q
├── scripts/
│   ├── eval_heldout.py             runner
│   └── redactors.py                target adapters
└── results/                        per-target per-question rows + bucket scores
    ├── 2026-05-02-caiioo-research-personal-data-detector-v14-pt.json
    ├── 2026-05-02-caiioo-research-personal-data-detector-v14-int8.json
    ├── 2026-05-02-openai-privacy-filter.json
    ├── 2026-05-02-google-gemma4-e2b.json
    └── 2026-05-02-qwen3.5-0.8b.json
```

Per-question feedback strings in the result JSONs include the gpt-4o judge's
one-sentence justification. These are useful for debugging individual
disagreements with the grade.

---

## Citation

```
@misc{privacybench_heldout_2026,
  title = {PII Redaction Held-Out Generalization Suite (2026)},
  author = {{Caiioo-ai}},
  howpublished = {\url{https://github.com/Caiioo-ai/privacybench}},
  year = {2026},
}
```

Maintained under MIT (see `../LICENSE`). PrivacyBench is a trademark of
Alex J. Wall (see `../NOTICE`). Six Cailloux, LLC.
