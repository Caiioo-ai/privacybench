# Codex low-effort results — 23 October 2026

These are **experimental results**, not replacements for the canonical PrivacyBench scores used by Caiioo's router. All seven models were run at explicit `low` effort through the official Codex app-server. The run did not preserve the original benchmark's model-specific effort settings: the existing Luna 6 and Luna 5.6 API runs used `xhigh`. Preserve one canonical score per model; this archive does not change that design or the canonical scoreboard.

## Results

| Runtime model | Effort | PB-NHS ↑ | 95% article-cluster interval |
|---|---|---:|---|
| gpt-6-astra | low | 97.63 | 95.35–99.57 |
| gpt-6.1-sol | low | 96.33 | 93.96–98.36 |
| gpt-6-sol | low | 96.31 | 93.72–98.43 |
| gpt-6-luna | low | 94.39 | 89.40–98.26 |
| gpt-5.6-luna | low | 87.75 | 83.85–91.26 |
| gpt-5.6-sol | low | 87.00 | 83.18–90.38 |
| gpt-5.6-terra | low | 86.72 | 82.68–89.97 |

The same ten articles, questions, prompts, frozen 26-item grounded prune, scoring functions and three-judge panel were used. Each model produced 165 QA responses and ten summaries: 175 inference calls and 149 scored evaluations. Total: 1,225 inference calls and 1,043 scored evaluations. Per-model intervals use 2,000 article-cluster resamples, seed 20261015.

PB-NHS is a weighted composite of grounded reading, near-miss abstention, closed-book abstention and summary fidelity. It is not the percentage of responses free from hallucination. All 14 component/composite metrics and intervals are in [summary.json](summary.json).

## Astra versus Sol 6.1 within this run

| Measurement | Astra | Sol 6.1 | Astra − Sol, paired 95% interval |
|---|---:|---:|---|
| PB-NHS ↑ | 97.63 | 96.33 | +1.30 [−0.02, +3.26] |
| Conditional QA hallucination rate ↓ | 1.11% | 2.25% | −1.14 pp [−3.43, 0.00] |
| Hallucinated summary claims ↓ | 0.85% | 1.59% | −0.74 pp [−2.37, +0.13] |
| False-answer rate ↓ | 1.04% | 2.08% | −1.04 pp [−3.15, 0.00] |

Astra has the better point estimates; the paired intervals do not establish decisive superiority. Grounded QA is identical: 42 correct answers, one partial and zero incorrect. The main composite difference is one avoided near-miss fabrication. A smaller contribution rewards Astra abstaining on a closed-book item that Sol answered correctly. Summary unsupported claims are 1/118 versus 2/126, with zero contradicted claims for either model; their claim sets differ. Summary fact coverage is 39/58 versus 38/58 and is not part of PB-NHS.

[paired-astra-sol.json](paired-astra-sol.json) uses the exact scorer with 10,000 paired article resamples, seed 20261023. Both models receive the same article draw. These are descriptive percentile intervals, not multiplicity-adjusted hypothesis tests. Ten articles, one response per item and fixed judge decisions limit inference. This comparison says nothing about Sol 6.1 at high effort or performance through other transports.

## Execution and grading

- Condition: `codex-app-server-isolated-v1`; run `20261023T153637920676Z`; CLI `codex-cli 0.161.0`.
- Inference: official Codex app-server using ChatGPT sign-in. Fresh ephemeral thread and empty working directory for each item; fixed base instructions; project instructions suppressed; tools, browsing, connectors, plugins and memory disabled; observed tool activity rejected. Codex still supplies runtime context.
- Exact runtime model acceptance is recorded. The upstream snapshot is unreported, so `model_returned` remains null. Temperature and maximum-output controls were not exposed by this adapter and remain null.
- Inference dollar cost is unknown, never zero. Measured tokens remain in every record. Cached input and reasoning tokens are subsets of input/output, not additive totals.
- Grading: deterministic prepass plus the existing independent `google/gemini-4-argon`, `meta/muse-wonder`, `x-ai/grok-4.7` panel via paid APIs. The original and paced recovery phases retain separate source hashes. Recovery completed the remaining four models without changing inference, judges, rubric or scoring.

The earlier API run used temperature 0 and different output controls. Its Sol 6.1 registry requested low, but saved telemetry records 169 calls with effort omitted and six with low; omitted does not establish the upstream effort. Luna 6 and Luna 5.6 record 175 xhigh calls each. The other four runtime names have no exact corresponding original result. Therefore this run is not an exact repetition of the API experiment. Corpus and grader reuse alone do not make execution conditions equal.

## Evidence and reproduction

The seven `*-graded.json` files retain every response, grade, judge vote, claim trace, metric, interval, token count and source hash. Public projections remove account allowance/funding snapshots, installed integration names and unpublished local file references. `summary.json` records both original and exported byte hashes and the transformations. The retained `publishable:false` field means the original canonical-router admission guard; it does not mean these research files are private.

Baseline corpus and scorer: repository commit `983c5c8d64a2a218e53e294710db621a58f16849`. This is a baseline reference, not a claim that the subsequently added Codex runner existed at that commit. Execution source hashes remain in each result. `scoring/rollup_and_plot.py` is a byte-exact snapshot of the baseline scorer; the frozen prune is [frozen-stage6.json](frozen-stage6.json).

From this directory, with Python 3 and no provider credentials:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 verify-results.py
PYTHONDONTWRITEBYTECODE=1 python3 paired-astra-sol.py \
  --astra-file gpt-6-astra-low-graded.json \
  --sol-file gpt-6.1-sol-low-graded.json \
  --prune-file summary.json --scorer scoring/rollup_and_plot.py \
  --output paired-astra-sol.json
```

Offline validation reproduces all 98 metric points and 98 confidence intervals, checks the seven exported file hashes, and preserves all 1,225 response/grading records. Paired analysis reproduces all 14 differences and intervals. No new inference or judge call is needed.

Publishing this archive does not import scores into Caiioo. Canonical updates require a consistent, explicit benchmark method, validation and the separate ingestion/rebuild step. No router or database update is included here.

Benchmark code, questions and results follow the repository's MIT license and existing corpus/prompt attribution in the [benchmark README](../../../README.md).
