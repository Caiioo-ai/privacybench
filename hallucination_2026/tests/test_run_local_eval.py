"""Tests for local OpenAI-compatible and Ollama PrivacyBench v2.1 runner."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_local_eval  # noqa: E402

MODEL = {
    "model_id": "apple/foundation-model",
    "display_name": "Apple Intelligence (On-Device)",
    "lab": "apple",
    "size_class": "local",
    "effective_cutoff": "2026-06-01",
    "cutoff_source": "vendor_published",
    "local": {
        "engine": "apple-foundation-models",
        "served_model": "apple-intelligence",
        "quantization": "on-device",
        "context_length": 8192,
    },
    "default_params": {
        "temperature": 0.0,
        "qa_max_tokens": 512,
        "summary_max_tokens": 768,
    },
}

ARTICLES = [
    {
        "id": "art1",
        "published_at": "2026-11-19",
        "domain": "privacy",
        "subtopic": "enforcement",
        "text_committed": True,
        "norm_text": "Article 1 states the fine was 42 million euros.",
    }
]

ITEMS = [
    {
        "id": "art1_q1",
        "article_id": "art1",
        "answerable": True,
        "domain": "privacy",
        "subtopic": "enforcement",
        "question": "How large was the fine?",
        "gold_answer": "42 million euros",
        "answer_type": "string",
        "numeric_precision_rule": None,
        "generator_model": "fixture",
        "modes": ["grounded_answerable", "closed_book"],
    },
    {
        "id": "art1_q2",
        "article_id": "art1",
        "answerable": False,
        "domain": "privacy",
        "subtopic": "enforcement",
        "question": "What was the CEO's salary?",
        "gold_answer": "NOT_ANSWERABLE",
        "answer_type": "unanswerable",
        "numeric_precision_rule": None,
        "generator_model": "fixture",
        "modes": ["grounded_nearmiss"],
    },
]


class LocalEvalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_strip_inline_think_preserves_final_answer_and_flags_unclosed_think(self):
        cleaned, had_think = run_local_eval.strip_inline_think(
            "<think>Let me check the article.\nThe fine is 42.</think>\n42 million euros"
        )
        self.assertEqual(cleaned, "42 million euros")
        self.assertTrue(had_think)

        # Unclosed <think> means the model ran out of tokens inside thinking; never leak raw CoT as answer.
        unclosed, had_unclosed = run_local_eval.strip_inline_think("<think>still reasoning forever")
        self.assertEqual(unclosed, "")
        self.assertTrue(had_unclosed)

    def test_strip_inline_think_handles_harmony_and_gemma_channel_tags(self):
        harmony = (
            "<|channel|>analysis<|message|>Check the figure.<|end|>"
            "<|start|>assistant<|channel|>final<|message|>42 million euros"
        )
        cleaned, had_think = run_local_eval.strip_inline_think(harmony)
        self.assertEqual(cleaned, "42 million euros")
        self.assertTrue(had_think)

        harmony_unclosed = "<|channel|>analysis<|message|>Still thinking without final channel..."
        cleaned_u, had_u = run_local_eval.strip_inline_think(harmony_unclosed)
        self.assertEqual(cleaned_u, "")
        self.assertTrue(had_u)

        gemma_chan = "<|channel>thought\nLet me check.<channel|>42 million euros"
        cleaned_g, had_g = run_local_eval.strip_inline_think(gemma_chan)
        self.assertEqual(cleaned_g, "42 million euros")
        self.assertTrue(had_g)

    def test_openai_compat_call_sends_no_auth_header_and_no_reasoning_effort(self):
        captured = {}

        class _FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "model": "apple-intelligence",
                        "choices": [
                            {
                                "message": {"role": "assistant", "content": "42 million euros"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 40, "completion_tokens": 5, "total_tokens": 45},
                    }
                ).encode("utf-8")

        def _fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["headers"] = {k.lower(): v for k, v in req.header_items()}
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeResp()

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            res = run_local_eval.call_local_endpoint(
                endpoint="http://127.0.0.1:18787/v1",
                wire="openai-chat",
                model_id="apple/foundation-model",
                served_model="apple-intelligence",
                prompt="How large was the fine?",
                temperature=0.0,
                max_tokens=512,
                timeout=60,
            )

        self.assertEqual(captured["url"], "http://127.0.0.1:18787/v1/chat/completions")
        self.assertNotIn("authorization", captured["headers"])
        self.assertNotIn("reasoning", captured["body"])
        self.assertNotIn("reasoning_effort", captured["body"])
        self.assertEqual(captured["body"]["model"], "apple-intelligence")
        self.assertEqual(res["content"], "42 million euros")
        self.assertEqual(res["cost_usd"], 0.0)
        self.assertEqual(res["model_requested"], "apple/foundation-model")
        self.assertEqual(res["finish_reason"], "stop")

    def test_ollama_chat_wire_sets_explicit_num_ctx(self):
        captured = {}

        class _FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "model": "gemma4:e2b",
                        "message": {"role": "assistant", "content": "Not stated in the article", "thinking": "checked"},
                        "done_reason": "stop",
                        "prompt_eval_count": 3800,
                        "eval_count": 120,
                    }
                ).encode("utf-8")

        def _fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeResp()

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            res = run_local_eval.call_local_endpoint(
                endpoint="http://127.0.0.1:11434",
                wire="ollama-chat",
                model_id="local/gemma4-e2b",
                served_model="gemma4:e2b",
                prompt="Question?",
                temperature=0.0,
                max_tokens=16384,
                timeout=300,
                num_ctx=32768,
            )

        self.assertEqual(captured["url"], "http://127.0.0.1:11434/api/chat")
        self.assertEqual(captured["body"]["options"]["num_ctx"], 32768)
        self.assertEqual(captured["body"]["options"]["num_predict"], 16384)
        self.assertEqual(res["content"], "Not stated in the article")
        self.assertEqual(res["prompt_tokens"], 3800)
        self.assertEqual(res["completion_tokens"], 120)
        self.assertEqual(res["cost_usd"], 0.0)

    def test_run_local_evaluation_writes_gradeable_schema_and_resumes_clean_cache(self):
        calls = []

        def _fake_call(**kw):
            calls.append(kw)
            # Fail the second call on the first run to prove resume only retries the failed item.
            if len(calls) == 2:
                raise RuntimeError("transient local timeout")
            return {
                "content": "42 million euros",
                "model_requested": kw["model_id"],
                "model_returned": kw["served_model"],
                "provider": "local:openai-chat",
                "reasoning_effort": None,
                "temperature": kw["temperature"],
                "prompt_tokens": 20,
                "completion_tokens": 5,
                "reasoning_tokens": 0,
                "total_tokens": 25,
                "cost_usd": 0.0,
                "latency_seconds": 0.1,
                "finish_reason": "stop",
            }

        with patch.object(run_local_eval, "call_local_endpoint", side_effect=_fake_call):
            out1 = run_local_eval.run_local_model_evaluation(
                copy.deepcopy(MODEL),
                copy.deepcopy(ARTICLES),
                copy.deepcopy(ITEMS),
                Path(self.tmp.name),
                run_date="2026-11-21",
                endpoint="http://127.0.0.1:18787/v1",
                wire="openai-chat",
                max_workers=1,
            )
            data1 = json.loads(out1.read_text(encoding="utf-8"))
            self.assertEqual(len(data1["qa_responses"]), 3)
            self.assertEqual(len(data1["summary_responses"]), 1)
            self.assertEqual(data1["inference_totals"]["total_cost_usd"], 0.0)
            self.assertEqual(len(calls), 4)

            # Second run should reuse the 3 clean calls and only execute the 1 failed call.
            out2 = run_local_eval.run_local_model_evaluation(
                copy.deepcopy(MODEL),
                copy.deepcopy(ARTICLES),
                copy.deepcopy(ITEMS),
                Path(self.tmp.name),
                run_date="2026-11-21",
                endpoint="http://127.0.0.1:18787/v1",
                wire="openai-chat",
                max_workers=1,
            )
            data2 = json.loads(out2.read_text(encoding="utf-8"))
            self.assertEqual(len(calls), 5)
            errors = [
                r
                for r in data2["qa_responses"] + data2["summary_responses"]
                if r["response"].startswith("[ERROR:")
            ]
            self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
