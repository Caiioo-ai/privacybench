# PII Redaction Task — 2026 Results

This directory contains 2026-vintage results on the canonical 25-question
PrivacyBench (`../evaluation_questions/privacybench_PII_redaction.json`).
The original 2024 results live in `../archive/`.

## Methodology

Identical to the original PrivacyBench methodology — see
`../README-redaction.md`. Same prompt, same 25 questions, same gpt-4o judge,
same 0–5 scoring with `(grade / 5.0) * 100` conversion. Numbers are
directly comparable to the 2024-era models in `../archive/`.

## Results — 2026 vintage

| Model | params | overall (%) | reproducible from |
|---|---:|---:|---|
| caiioo-research/personal-data-detector-v14 | 117M | **91.2** | `2026-05-02-caiioo-research-personal-data-detector-v14.json` (per-question scores) |

Selected reference points from `../archive/` (2024 numbers, same questions):

| Model | params | overall (%) |
|---|---:|---:|
| gpt-4o | (closed) | 97.6 |
| gemma2:9b | 9B | 92.8 |
| qwen2.5:7b | 7B | 91.2 |
| openai/privacy-filter (token-class, no judge) | 50M | 87.2 |
| mistral-latest | 7B | 85.6 |
| qwen2.5:3b | 3B | 83.2 |
| llama3.2:3b | 3B | 78.4 |
| deepseek-r1:7b | 7B | 78.4 |
| gemma2:2b | 2B | 76.0 |
| llama3.2:1b | 1B | 0.8 |

(See `../privacybench_score_report-*.json` for full per-question breakdowns.)

## Notes on the 2026 entry

`caiioo-research/personal-data-detector-v14` is a token-classification model
(`intfloat/multilingual-e5-small` backbone, 117M params, ~113 MB int8 ONNX
when deployed). Like `openai/privacy-filter`, it produces per-token
predictions which are then mapped to bracketed placeholders, rather than
generating redacted text directly. The judge scores the resulting redacted
text identically to the generative-model entries in `../archive/`.

Model weights are not published with this benchmark entry — the
`caiioo-research/personal-data-detector-v14` identifier refers to a
proprietary model owned by Six Cailloux, LLC. (see the repo-root
`MODELS.md` for the full ownership / availability disclosure across all
referenced models). The per-question result file in this directory is
reproducible by anyone with a HuggingFace-compatible token-classification
model:

```bash
cd ../../pii_redaction_heldout_2026/scripts
python eval_heldout.py --target hf-token-class:<your_repo_id>
```

The same sister directory's `eval_heldout.py` runner can also be pointed at
the canonical 25-question PrivacyBench question file via a small loader
patch, but the canonical task currently uses
`../benchmark-list-of-models.py` and `../benchmark-single-model.py` for
its own runs.

## Cross-task: what the held-out generalization suite says about this same model

The 117M-param caiioo-research entry above scores 91.2 on the canonical
25-question PrivacyBench (this task). On the harder 150-question held-out
generalization suite (`../../pii_redaction_heldout_2026/`) — which tests
dense formal documents, 11 languages, code-paste with embedded secrets, and
negative-control prose — the same model scores 87.57 (PT) / 85.90 (int8
ONNX). See that task's README for the full per-bucket table including
openai/privacy-filter, gemma4:e2b, and qwen3.5:0.8b on the same set.
