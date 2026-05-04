"""
Redactor adapters — turn `--target <string>` into a callable
`(text: str) -> str` that produces a redacted version of the input.

Add a new model by writing an adapter here and updating `build_redactor`.
"""
from __future__ import annotations

import re
from typing import Callable

import requests


# ─── openai/privacy-filter (token-classification) ─────────────────────────

# Mapping privacy-filter labels → bracketed placeholder text used by the
# original PrivacyBench leaderboard prompt convention.
PF_PLACEHOLDER_MAP = {
    "private_person": "[NAME]",
    "private_email": "[EMAIL]",
    "private_phone": "[PHONE]",
    "private_address": "[ADDRESS]",
    "private_url": "[URL]",
    "private_date": "[DATE]",
    "account_number": "[ACCOUNT_NUMBER]",
    "secret": "[SECRET]",
}

# Regex augment used by the privacy-filter+regex variant — covers
# identifiers privacy-filter's 8 categories don't explicitly handle but
# PrivacyBench includes (SSN, passport, DL, IP, biometric IDs).
PF_REGEX_AUGMENTS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[IP_ADDRESS]"),
    (re.compile(r"\b[A-Z]\d{7,9}\b"), "[PASSPORT_NUMBER]"),
    (re.compile(r"\bD\d{7,9}\b"), "[DRIVERS_LICENSE]"),
    (re.compile(r"\bFP-[A-Z0-9]+\b"), "[BIOMETRIC_ID]"),
    (re.compile(r"\bFR-[A-Z0-9]+\b"), "[BIOMETRIC_ID]"),
    (re.compile(r"\bU\d{6,9}\b"), "[UNIVERSITY_ID]"),
]


def _build_privacy_filter(use_regex: bool) -> Callable[[str], str]:
    from transformers import pipeline

    pipe = pipeline(
        task="token-classification",
        model="openai/privacy-filter",
        aggregation_strategy="simple",
        device_map="auto",
    )

    def _trim_span(text, start, end):
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return start, end

    def detect(text: str):
        raw = pipe(text)
        spans = []
        for ent in raw:
            s, e = _trim_span(text, int(ent["start"]), int(ent["end"]))
            if s >= e:
                continue
            spans.append((s, e, ent["entity_group"], float(ent["score"])))
        spans.sort(key=lambda s: s[0])
        # Merge consecutive same-label whitespace-separated spans.
        merged = []
        for s, e, label, score in spans:
            if merged:
                ps, pe, plabel, _ = merged[-1]
                if label == plabel and text[pe:s].strip() == "":
                    merged[-1] = (ps, e, label, score)
                    continue
            merged.append((s, e, label, score))
        return merged

    def regex_extra(text: str):
        spans = []
        for pattern, label in PF_REGEX_AUGMENTS:
            for m in pattern.finditer(text):
                spans.append((m.start(), m.end(), label, 1.0))
        return spans

    def merge_spans(model_spans, regex_spans):
        # Regex wins on overlap (placeholder is more specific).
        tagged = (
            [(s, e, l, sc, "model") for s, e, l, sc in model_spans]
            + [(s, e, l, sc, "regex") for s, e, l, sc in regex_spans]
        )
        tagged.sort(key=lambda s: (s[0], 0 if s[4] == "regex" else 1, -(s[1] - s[0])))
        out = []
        for span in tagged:
            s, e = span[0], span[1]
            if not out:
                out.append(span)
                continue
            ps, pe = out[-1][0], out[-1][1]
            if s < pe:
                if span[4] == "regex" and out[-1][4] == "model":
                    out[-1] = span
                elif span[4] == out[-1][4] and (e - s) > (pe - ps):
                    out[-1] = span
            else:
                out.append(span)
        return [(s, e, l, sc) for (s, e, l, sc, _) in out]

    def apply_redaction(text, spans):
        out = text
        for start, end, label, _ in sorted(spans, key=lambda s: -s[0]):
            placeholder = (
                label if isinstance(label, str) and label.startswith("[")
                else PF_PLACEHOLDER_MAP.get(label, f"[{label.upper()}]")
            )
            out = out[:start] + placeholder + out[end:]
        return out

    def redact(text: str) -> str:
        spans = detect(text)
        if use_regex:
            spans = merge_spans(spans, regex_extra(text))
        return apply_redaction(text, spans)

    return redact


# ─── Ollama generative ────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_PROMPT = (
    "Examine the following text excerpt and redact all information that "
    "you would associate with an identified or identifiable individual "
    "human and therefore qualifies as personal data. For redaction, "
    "replace each piece of personal data with a placeholder that "
    "indicates the type of data has been removed (e.g. [NAME], [EMAIL], "
    "[PHONE], [ADDRESS], [DATE], [ID], [SECRET]). Reply ONLY with the "
    "redacted text, no commentary.\n\nText:\n{text}\n\nRedacted text:"
)


def _build_ollama(model_id: str) -> Callable[[str], str]:
    def redact(text: str) -> str:
        payload = {
            "model": model_id,
            "prompt": OLLAMA_PROMPT.format(text=text),
            "stream": False,
            "think": False,
        }
        r = requests.post(OLLAMA_URL, json=payload, timeout=600.0)
        r.raise_for_status()
        out = r.json().get("response", "").strip()
        if out.lower().startswith(("here", "sure", "redacted text:", "okay")):
            i = out.find("\n")
            if i != -1:
                out = out[i + 1:].strip()
        return out

    return redact


# ─── HuggingFace token-classification (any model with the token-class head) ─

# Mapping label → bracketed placeholder. Defaults map to the
# PrivacyBench convention; unknown labels become `[<UPPER>]`.
HF_TC_PLACEHOLDER_MAP = {
    "person_name": "[NAME]",
    "email_address": "[EMAIL]",
    "phone_or_fax": "[PHONE]",
    "postal_address": "[ADDRESS]",
    "url": "[URL]",
    "birth_date": "[DATE]",
    "government_id": "[ID]",
    "institutional_id": "[ID]",
    "financial_account": "[PAYMENT]",
    "authentication_secret": "[SECRET]",
    "online_handle": "[HANDLE]",
    "ip_address": "[IP_ADDRESS]",
    "device_id": "[DEVICE_ID]",
    "vehicle_id": "[VEHICLE_ID]",
    "biometric_id": "[BIOMETRIC_ID]",
    "precise_geolocation": "[GEOLOCATION]",
}


def _build_hf_token_class(repo_id: str) -> Callable[[str], str]:
    """Generic HF token-classification adapter. Reads the model's config
    for id2label, decodes BIOES tags, applies bracketed placeholders."""
    import numpy as np
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForTokenClassification.from_pretrained(repo_id)
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    def predict_spans(text: str):
        import torch
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

        spans = []
        cur = None
        for tag, off in zip(tags, offsets):
            cs, ce = int(off[0]), int(off[1])
            if cs == 0 and ce == 0:
                continue
            if tag == "O":
                if cur is not None:
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

        # Trim whitespace + merge same-label whitespace-only gaps.
        out = []
        for sp in spans:
            s, e = sp["start"], sp["end"]
            while s < e and text[s].isspace():
                s += 1
            while e > s and text[e - 1].isspace():
                e -= 1
            if s < e:
                out.append({"start": s, "end": e, "label": sp["label"]})
        merged = []
        for sp in out:
            if merged and sp["label"] == merged[-1]["label"] and text[merged[-1]["end"]:sp["start"]].strip() == "":
                merged[-1] = {"start": merged[-1]["start"], "end": sp["end"], "label": sp["label"]}
                continue
            merged.append(sp)
        return merged

    def redact(text: str) -> str:
        spans = predict_spans(text)
        out = text
        for sp in sorted(spans, key=lambda x: -x["start"]):
            ph = HF_TC_PLACEHOLDER_MAP.get(sp["label"], f"[{sp['label'].upper()}]")
            out = out[:sp["start"]] + ph + out[sp["end"]:]
        return out

    return redact


# ─── Dispatch ─────────────────────────────────────────────────────────────

def build_redactor(target: str) -> Callable[[str], str]:
    if target == "privacy-filter":
        return _build_privacy_filter(use_regex=False)
    if target == "privacy-filter-regex":
        return _build_privacy_filter(use_regex=True)
    if target.startswith("ollama:"):
        return _build_ollama(target[len("ollama:"):])
    if target.startswith("hf-token-class:"):
        return _build_hf_token_class(target[len("hf-token-class:"):])
    raise ValueError(
        f"Unknown target: {target!r}. Supported: 'privacy-filter', "
        "'privacy-filter-regex', 'ollama:<model>', 'hf-token-class:<repo>'."
    )
