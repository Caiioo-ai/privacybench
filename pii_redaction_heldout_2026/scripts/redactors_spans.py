"""
Span-level predictor adapters — return list of {start, end, label} dicts
instead of a redacted text string.

Used by `hipaa_lds_task/scripts/eval_hipaa_lds.py`, which needs span
boundaries to compare against expected PHI spans + carve-outs.

Mirrors the dispatch convention of `redactors.py` (text-out adapters).
"""
from __future__ import annotations

from typing import Callable, List, TypedDict


class Span(TypedDict):
    start: int
    end: int
    label: str


# ─── openai/privacy-filter (token-classification) ─────────────────────────

def _build_privacy_filter_spans(use_regex: bool) -> Callable[[str], List[Span]]:
    import re
    from transformers import pipeline

    pipe = pipeline(
        task="token-classification",
        model="openai/privacy-filter",
        aggregation_strategy="simple",
        device_map="auto",
    )

    # Map privacy-filter labels to LDS-comparable labels for class-accuracy
    # scoring. The carve-out logic is span-only (whether predicted spans
    # OVERLAP retained signals), so this mapping isn't load-bearing — it
    # just sets the predicted .label for the rubric's class-accuracy bonus.
    PF_TO_LDS = {
        "private_person": "name",
        "private_email": "email",
        "private_phone": "phone_or_fax",
        "private_address": "street_address",
        "private_url": "url",
        "private_date": "date",
        "account_number": "account_number",
        "secret": "auth_secret",
    }

    REGEX_AUGMENTS = [
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "ssn"),
        (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "ip_address"),
        (re.compile(r"\b[A-Z]\d{7,9}\b"), "passport"),
        (re.compile(r"\bD\d{7,9}\b"), "drivers_license"),
        (re.compile(r"\bFP-[A-Z0-9]+\b"), "biometric_id"),
        (re.compile(r"\bFR-[A-Z0-9]+\b"), "biometric_id"),
        (re.compile(r"\bU\d{6,9}\b"), "university_id"),
    ]

    def _trim(text, s, e):
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        return s, e

    def predict(text: str) -> List[Span]:
        raw = pipe(text)
        spans: List[Span] = []
        for ent in raw:
            s, e = _trim(text, int(ent["start"]), int(ent["end"]))
            if s >= e:
                continue
            spans.append({
                "start": s,
                "end": e,
                "label": PF_TO_LDS.get(ent["entity_group"], ent["entity_group"]),
            })
        spans.sort(key=lambda x: x["start"])
        # Merge consecutive same-label whitespace-only gaps.
        merged: List[Span] = []
        for sp in spans:
            if (
                merged
                and sp["label"] == merged[-1]["label"]
                and text[merged[-1]["end"]:sp["start"]].strip() == ""
            ):
                merged[-1] = {
                    "start": merged[-1]["start"],
                    "end": sp["end"],
                    "label": sp["label"],
                }
                continue
            merged.append(sp)
        if use_regex:
            for pattern, label in REGEX_AUGMENTS:
                for m in pattern.finditer(text):
                    merged.append({"start": m.start(), "end": m.end(), "label": label})
            merged.sort(key=lambda x: (x["start"], -x["end"]))
            # Drop overlaps; longer / earlier wins.
            out: List[Span] = []
            for sp in merged:
                if out and sp["start"] < out[-1]["end"]:
                    continue
                out.append(sp)
            return out
        return merged

    return predict


# ─── Generic HF token-classification ──────────────────────────────────────

def _build_hf_token_class_spans(repo_id: str) -> Callable[[str], List[Span]]:
    """Generic HF token-classification adapter returning BIOES-decoded
    spans. Reads model.config.id2label for the label space."""
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForTokenClassification.from_pretrained(repo_id)
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    def predict(text: str) -> List[Span]:
        enc = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            logits = model(**enc).logits[0]
        pred_ids = logits.argmax(dim=-1).tolist()
        tags = [id2label.get(i, "O") for i in pred_ids]

        spans: List[Span] = []
        cur = None
        for tag, off in zip(tags, offsets):
            cs, ce = int(off[0]), int(off[1])
            if cs == 0 and ce == 0:
                continue
            if tag == "O":
                if cur:
                    spans.append(cur)
                    cur = None
                continue
            prefix, _, label = tag.partition("-")
            if prefix == "S":
                if cur:
                    spans.append(cur)
                spans.append({"start": cs, "end": ce, "label": label})
                cur = None
            elif prefix == "B":
                if cur:
                    spans.append(cur)
                cur = {"start": cs, "end": ce, "label": label}
            elif prefix == "I":
                if cur and cur["label"] == label:
                    cur["end"] = ce
                else:
                    if cur:
                        spans.append(cur)
                    cur = {"start": cs, "end": ce, "label": label}
            elif prefix == "E":
                if cur and cur["label"] == label:
                    cur["end"] = ce
                    spans.append(cur)
                    cur = None
                else:
                    if cur:
                        spans.append(cur)
                    spans.append({"start": cs, "end": ce, "label": label})
                    cur = None
        if cur:
            spans.append(cur)

        # Trim whitespace + merge whitespace-only gaps within same label.
        out: List[Span] = []
        for sp in spans:
            s, e = sp["start"], sp["end"]
            while s < e and text[s].isspace():
                s += 1
            while e > s and text[e - 1].isspace():
                e -= 1
            if s < e:
                out.append({"start": s, "end": e, "label": sp["label"]})
        merged: List[Span] = []
        for sp in out:
            if (
                merged
                and sp["label"] == merged[-1]["label"]
                and text[merged[-1]["end"]:sp["start"]].strip() == ""
            ):
                merged[-1] = {"start": merged[-1]["start"], "end": sp["end"], "label": sp["label"]}
                continue
            merged.append(sp)
        return merged

    return predict


# ─── Dispatch ─────────────────────────────────────────────────────────────

def build_span_predictor(target: str) -> Callable[[str], List[Span]]:
    if target == "privacy-filter":
        return _build_privacy_filter_spans(use_regex=False)
    if target == "privacy-filter-regex":
        return _build_privacy_filter_spans(use_regex=True)
    if target.startswith("hf-token-class:"):
        return _build_hf_token_class_spans(target[len("hf-token-class:"):])
    raise ValueError(
        f"Unknown target: {target!r}. Supported: 'privacy-filter', "
        "'privacy-filter-regex', 'hf-token-class:<repo>'. "
        "Generative-model targets (Ollama) cannot be span-evaluated — "
        "use eval_heldout.py for those."
    )
