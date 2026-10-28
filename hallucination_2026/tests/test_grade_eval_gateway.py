"""Judges on CometAPI: each asks its own snapped effort and gets enough attempts to outlast refused channels.

2026-10-28: with 4 attempts, a ling-3.1-flash regrade died on gpt-6.1-sol served four times as igpt-6_1-sol, and a
mimo-v2.6-pro regrade on gemini-4-argon served four times with no model name; the grader writes only at the end, so
each crash discarded the file's paid grading.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import grade_eval


class JudgeGateway(unittest.TestCase):
    def setUp(self):
        self.saved = (dict(grade_eval.JUDGE_EFFORT), dict(grade_eval.JUDGE_TIMEOUT), grade_eval.JUDGE_RETRIES)
        self.addCleanup(self.restore)

    def restore(self):
        grade_eval.JUDGE_EFFORT.clear(); grade_eval.JUDGE_EFFORT.update(self.saved[0])
        grade_eval.JUDGE_TIMEOUT.clear(); grade_eval.JUDGE_TIMEOUT.update(self.saved[1])
        grade_eval.JUDGE_RETRIES = self.saved[2]
        import llm_client
        llm_client._GATEWAY.update(name='openrouter', routes={})

    def test_cometapi_gives_judges_more_attempts_and_their_own_efforts(self):
        catalog = {'data': [{'id': jm, 'supported_parameters': ['reasoning', 'reasoning_effort'],
                             'reasoning': {'supported_efforts': ['high', 'low']}} for jm in grade_eval.JUDGE_PANEL[:2]]
                   + [{'id': grade_eval.JUDGE_PANEL[2], 'supported_parameters': ['reasoning'], 'reasoning': {'mandatory': True}}]}
        with patch.object(grade_eval, 'cometapi_routes', return_value={}), \
             patch.object(grade_eval.cometapi_client, 'fetch_catalog', return_value={}), \
             patch.object(grade_eval, 'fetch_capability_catalog', return_value=catalog):
            grade_eval.use_gateway('cometapi', Path('unused'))
        self.assertGreaterEqual(grade_eval.JUDGE_RETRIES, 8)
        self.assertEqual(grade_eval.JUDGE_EFFORT[grade_eval.JUDGE_PANEL[2]], None)

    def test_judge_calls_carry_the_attempts(self):
        grade_eval.JUDGE_RETRIES = 9
        seen = {}
        def fake(**kw):
            seen.setdefault('max_retries', set()).add(kw.get('max_retries'))
            return {'content': '{"grade": "CORRECT"}', 'model_requested': kw['model_id'], 'cost_usd': 0.0}
        item = {'item_id': 'i', 'question': 'q', 'gold_answer': 'a', 'response': 'b', 'answer_type': 'string',
                'answerable': True, 'eval_mode': 'grounded_answerable', 'numeric_precision_rule': None}
        with patch.object(grade_eval, 'call_llm', side_effect=fake):
            grade_eval.judge_qa_with_panel(item)
        self.assertEqual(seen['max_retries'], {9})


if __name__ == '__main__':
    unittest.main()
