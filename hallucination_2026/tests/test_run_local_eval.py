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

    def test_detect_generation_loop_matches_caiioo_patterns_without_false_positives(self):
        # Pattern 1: identical character run (128+)
        self.assertEqual(run_local_eval.detect_generation_loop("x" * 130), "identical_char_run")

        # Pattern 2: Caiioo intra-message phrase repetition
        phrase_loop = "Wait, let me re-read the article carefully to verify. " * 4
        self.assertEqual(run_local_eval.detect_generation_loop(phrase_loop), "intra_message_repetition")

        # Normal markdown table & URLs must NOT false-positive
        table = (
            "| Metric | Score | Notes |\n"
            "| --- | --- | --- |\n"
            "| Accuracy | 95% | Clean |\n"
            "| Abstention | 90% | [1](https://example.com/very/long/citation/path/here) |\n"
            "| Summary | 98% | [2](https://example.com/very/long/citation/path/here) |\n"
            "| Recall | 92% | [3](https://example.com/very/long/citation/path/here) |\n"
        )
        self.assertIsNone(run_local_eval.detect_generation_loop(table))

        normal = (
            "The article reports a 42 million euro fine imposed on November 19, 2026. "
            "It does not mention the CEO's salary or any executive compensation figures."
        )
        self.assertIsNone(run_local_eval.detect_generation_loop(normal))

    def test_ollama_stream_aborts_immediately_on_thinking_loop_with_finish_reason_loop(self):
        chunks_read = 0
        repeated_unit = "Wait, let me double-check whether the article states the exact figure. "

        class _FakeStreamingResp:
            def __init__(self):
                self.closed = False
                self._lines = [
                    json.dumps({"model": "qwen3.5:2b", "message": {"role": "assistant", "thinking": repeated_unit}, "done": False}).encode("utf-8") + b"\n"
                    for _ in range(50)
                ] + [
                    json.dumps({"model": "qwen3.5:2b", "message": {"role": "assistant", "content": "never reached"}, "done": True, "done_reason": "stop", "eval_count": 9999}).encode("utf-8") + b"\n"
                ]

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.closed = True
                return False

            def close(self):
                self.closed = True

            def __iter__(self):
                nonlocal chunks_read
                for line in self._lines:
                    chunks_read += 1
                    yield line

        fake_resp = _FakeStreamingResp()
        with patch("urllib.request.urlopen", return_value=fake_resp):
            res = run_local_eval.call_local_endpoint(
                endpoint="http://127.0.0.1:11434",
                wire="ollama-chat",
                model_id="local/qwen3.5-2b-q4_k_m",
                served_model="qwen3.5:2b-q4_K_M",
                prompt="Question?",
                temperature=0.0,
                max_tokens=32768,
                timeout=300,
                num_ctx=32768,
            )

        self.assertEqual(res["finish_reason"], "loop")
        self.assertTrue(res["loop_detected"])
        self.assertEqual(res["loop_channel"], "thinking")
        self.assertEqual(res["content"], "")
        self.assertTrue(res["reasoned"])
        self.assertLess(chunks_read, 15)

    def test_ollama_stream_logs_tokens_per_second_and_decode_tps(self):
        lines = [
            json.dumps({"model": "gemma4:e2b", "message": {"role": "assistant", "thinking": "checked. "}, "done": False}).encode("utf-8") + b"\n",
            json.dumps({"model": "gemma4:e2b", "message": {"role": "assistant", "content": "42 million euros"}, "done": False}).encode("utf-8") + b"\n",
            json.dumps({
                "model": "gemma4:e2b",
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3000,
                "prompt_eval_duration": 500_000_000,
                "eval_count": 100,
                "eval_duration": 1_000_000_000,
            }).encode("utf-8") + b"\n",
        ]

        class _FakeStream:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def __iter__(self):
                return iter(lines)

        with patch("urllib.request.urlopen", return_value=_FakeStream()):
            res = run_local_eval.call_local_endpoint(
                endpoint="http://127.0.0.1:11434",
                wire="ollama-chat",
                model_id="local/gemma4-e2b",
                served_model="gemma4:e2b",
                prompt="Question?",
                temperature=0.0,
                max_tokens=32768,
                timeout=300,
                num_ctx=32768,
            )

        self.assertEqual(res["finish_reason"], "stop")
        self.assertEqual(res["content"], "42 million euros")
        self.assertEqual(res["decode_tokens_per_second"], 100.0)
        self.assertGreater(res["tokens_per_second"], 0.0)

    def test_grade_eval_and_rollup_treat_loop_as_distinct_end_reason_not_abstention(self):
        import grade_eval  # noqa: E402
        import rollup_and_plot  # noqa: E402

        loop_qa = {
            "item_id": "art1_q2",
            "article_id": "art1",
            "eval_mode": "grounded_nearmiss",
            "answerable": False,
            "question": "What was the CEO's salary?",
            "gold_answer": "NOT_ANSWERABLE",
            "response": "",
            "telemetry": {"finish_reason": "loop", "loop_detected": True},
        }
        pre = grade_eval.deterministic_prepass(loop_qa)
        self.assertIsNotNone(pre)
        self.assertEqual(pre["grading_path"], "deterministic_loop")
        self.assertEqual(pre["final_grade"], "LOOP")

        # Loop on unanswerable/closed_book must NOT earn free abstention credit in rollup
        qa_items = [
            {**loop_qa, "grading": pre},
            {
                "item_id": "art1_q1",
                "article_id": "art1",
                "eval_mode": "grounded_answerable",
                "answerable": True,
                "response": "",
                "telemetry": {"finish_reason": "loop"},
                "grading": {"grading_path": "deterministic_loop", "final_grade": "LOOP", "hedged": False},
            },
            {
                "item_id": "art1_q1",
                "article_id": "art1",
                "eval_mode": "closed_book",
                "answerable": True,
                "response": "",
                "telemetry": {"finish_reason": "loop"},
                "grading": {"grading_path": "deterministic_loop", "final_grade": "LOOP", "hedged": False},
            },
        ]
        metrics = rollup_and_plot.compute_metrics_from_subsets(qa_items, [])
        self.assertEqual(metrics["nearmiss_abstention_pct"], 0.0)
        self.assertEqual(metrics["closed_book_abstention_pct"], 0.0)
        self.assertEqual(metrics["grounded_accuracy_pct"], 0.0)
        self.assertEqual(metrics["loop_count"], 3)

    def test_grade_eval_treats_high_cap_empty_length_as_dnf_while_still_blocking_low_cap(self):
        import grade_eval  # noqa: E402
        import rollup_and_plot  # noqa: E402

        dnf_qa = {
            "item_id": "art1_q2",
            "article_id": "art1",
            "eval_mode": "grounded_nearmiss",
            "answerable": False,
            "max_tokens": 32768,
            "question": "What was the CEO's salary?",
            "gold_answer": "NOT_ANSWERABLE",
            "response": "",
            "telemetry": {"finish_reason": "length", "completion_tokens": 32768},
        }
        pre = grade_eval.deterministic_prepass(dnf_qa)
        self.assertIsNotNone(pre)
        self.assertEqual(pre["grading_path"], "deterministic_dnf")
        self.assertEqual(pre["final_grade"], "DNF")

        metrics = rollup_and_plot.compute_metrics_from_subsets([{**dnf_qa, "grading": pre}], [])
        self.assertEqual(metrics["nearmiss_abstention_pct"], 0.0)
        self.assertEqual(metrics["dnf_count"], 1)

        # Low-cap empty length (e.g. hosted max_tokens=512) must STILL be refused by grade_result_file
        low_cap_path = Path(self.tmp.name) / "low-cap.json"
        low_cap_path.write_text(json.dumps({
            "model_meta": {"model_id": "test/low", "default_params": {"qa_max_tokens": 512, "summary_max_tokens": 768}},
            "qa_responses": [{**dnf_qa, "max_tokens": 512, "telemetry": {"finish_reason": "length", "completion_tokens": 512}}],
            "summary_responses": [],
        }), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            grade_eval.grade_result_file(low_cap_path, {"art1": ARTICLES[0]}, max_workers=1)

        # High-cap empty length (>=16384) must grade cleanly without calling judges
        high_cap_path = Path(self.tmp.name) / "high-cap.json"
        high_cap_path.write_text(json.dumps({
            "model_meta": {"model_id": "local/high", "default_params": {"qa_max_tokens": 32768, "summary_max_tokens": 32768}},
            "qa_responses": [dnf_qa],
            "summary_responses": [],
        }), encoding="utf-8")
        grade_eval.grade_result_file(high_cap_path, {"art1": ARTICLES[0]}, max_workers=1)
        graded = json.loads(high_cap_path.read_text(encoding="utf-8"))
        self.assertEqual(graded["qa_responses"][0]["grading"]["final_grade"], "DNF")


if __name__ == "__main__":
    unittest.main()
