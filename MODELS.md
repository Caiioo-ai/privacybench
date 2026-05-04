# About the models referenced in PrivacyBench results

**Short version:** the *benchmark* is MIT (see `LICENSE`). Specific *models*
appearing in result tables are owned by their respective publishers and
licensed separately — including the `caiioo-research/*` models authored by
Six Cailloux, LLC, which are **proprietary and not currently published.**

This file exists so a reader of the result tables understands what is and is
not openly available, what they can and cannot use freely, and where to go
for clarification.

---

## Models referenced in result tables

The result JSONs and READMEs in this repository reference models from several
publishers. Each is governed by its own license terms. PrivacyBench itself
does not redistribute any model weights.

### Openly available (downloadable, license terms apply)

| Identifier in results | Publisher | Where to obtain | License |
|---|---|---|---|
| `openai/privacy-filter` | OpenAI | huggingface.co/openai/privacy-filter | OpenAI's published terms |
| `google/gemma2:9b`, `google/gemma2:2b` | Google | ollama.com/library/gemma2 (and HF) | Gemma Terms of Use |
| `google/gemma4:e2b` | Google | ollama.com/library/gemma4 | Gemma Terms of Use |
| `qwen2.5:7b`, `qwen2.5:3b`, `qwen3.5:0.8b` | Alibaba | ollama.com/library/qwen2.5 (and HF) | Qwen License |
| `mistral-latest` | Mistral AI | ollama.com/library/mistral | Apache-2.0 (model-dependent) |
| `llama3.2:3b`, `llama3.2:1b` | Meta | ollama.com/library/llama3.2 | Llama 3.2 Community License |
| `deepseek-r1:7b` | DeepSeek | ollama.com/library/deepseek-r1 | DeepSeek License |

### Closed (not publicly available, used as reference only)

| Identifier in results | Publisher | Notes |
|---|---|---|
| `gpt-4o` | OpenAI | API-only, used here as a high-end reference point |

### Proprietary — Six Cailloux / Caiioo (not published)

| Identifier in results | Publisher | Status |
|---|---|---|
| `caiioo-research/personal-data-detector-v14` | Six Cailloux, LLC | **Proprietary. Not published.** Trained-weight artifacts and training corpus are not distributed with this repository, are not on the HuggingFace public model registry, and are not available for download. The identifier is an internal version label, not a downloadable repository. |
| `caiioo-research/hipaa-lds-detector` | Six Cailloux, LLC | **Proprietary. Not published.** Same caveats as above. |

**What you CAN do with the `caiioo-research/*` entries in the results:**

- Cite the published numbers (with source attribution to this repository).
- Independently reproduce comparable scores by training your own
  token-classification model and running it through the same evaluation
  scripts (`pii_redaction_heldout_2026/scripts/eval_heldout.py` accepts any
  HuggingFace token-classification repo via `--target hf-token-class:<id>`).
- Contest the methodology, propose corrections, or contribute additional
  test questions via PR.

**What you CANNOT do with the `caiioo-research/*` entries:**

- Download the weights — they are not published.
- Republish the published numbers as your own model's scores.
- Imply that a third-party model with a similar name is the same model
  ("personal-data-detector", "hipaa-lds-detector", and the version
  identifier `v14` together are reserved Six Cailloux internal labels for
  these specific trained artifacts).

---

## Trademark and product attribution

* **PrivacyBench** is a trademark of Alex J. Wall (see `NOTICE`).
* **Caiioo**, **Six Cailloux**, and the model name conventions
  `caiioo-research/personal-data-detector-*` and
  `caiioo-research/hipaa-lds-detector` refer to proprietary models built
  and operated by Six Cailloux, LLC.
* The `caiioo-research/*` model identifiers are not generic — they refer to
  specific trained checkpoints whose weights and training corpus are
  Six Cailloux, LLC. trade secrets.

---

## Reproducibility versus replication

PrivacyBench is designed to make *evaluation* reproducible — every question,
every per-question score, every grading prompt, and every adapter is in
this repository, so anyone can run a different model through the same
pipeline and produce a directly comparable number.

PrivacyBench is **not** designed to make *model replication* possible.
Reproducing the `caiioo-research/*` results would require independently
training a model on a comparable corpus — which is the work of model
construction, not benchmark execution. The benchmark publishes the test, not
the model under test.

---

## Contact

For licensing questions about the proprietary models referenced in this
repository, or to request commercial use of the `caiioo-research/*` models
in a product, contact Six Cailloux, LLC. via caiioo.ai.

Questions about the benchmark itself (questions, methodology, scoring) can
be raised as GitHub issues on this repository.
