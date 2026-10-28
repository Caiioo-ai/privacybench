"""Judges on CometAPI: each asks its own snapped effort and gets enough attempts to outlast refused channels.

2026-10-28: with 4 attempts, a ling-3.1-flash regrade died on gpt-6.1-sol served four times as igpt-6_1-sol, and a
mimo-v2.6-pro regrade on gemini-4-argon served four times with no model name; the grader writes only at the end, so
each crash discarded the file's paid grading.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import grade_eval


class Panel(unittest.TestCase):
    def setUp(self):
        self.saved = (dict(grade_eval.JUDGE_EFFORT), dict(grade_eval.JUDGE_TIMEOUT), grade_eval.JUDGE_RETRIES,
                      dict(grade_eval.JUDGE_GATEWAY), dict(grade_eval.JUDGE_ROUTE_NOTES))
        self.addCleanup(self.restore)

    def restore(self):
        grade_eval.JUDGE_EFFORT.clear(); grade_eval.JUDGE_EFFORT.update(self.saved[0])
        grade_eval.JUDGE_TIMEOUT.clear(); grade_eval.JUDGE_TIMEOUT.update(self.saved[1])
        grade_eval.JUDGE_RETRIES = self.saved[2]
        grade_eval.JUDGE_GATEWAY.clear(); grade_eval.JUDGE_GATEWAY.update(self.saved[3])
        grade_eval.JUDGE_ROUTE_NOTES.clear(); grade_eval.JUDGE_ROUTE_NOTES.update(self.saved[4])
        import llm_client
        llm_client._GATEWAY.update(name='openrouter', routes={})

    def registry(self, **entries):
        """A models.json whose judge entries are given; the rest of the panel has a CometAPI route."""
        models = [entries.get(jm, {'cometapi': {'model': jm.split('/')[1], 'wire': 'openai-chat'}}) | {'model_id': jm}
                  for jm in grade_eval.JUDGE_PANEL]
        path = Path(tempfile.mkdtemp()) / 'models.json'
        path.write_text(json.dumps({'models': models}))
        return path

    def use_cometapi_panel(self, models_file, catalog=None):
        asked = {}
        def routes(models_file, model_ids, catalog):
            asked['models'] = list(model_ids)
            return {m: {} for m in model_ids}
        with patch.object(grade_eval, 'cometapi_routes', side_effect=routes), \
             patch.object(grade_eval.cometapi_client, 'fetch_catalog', return_value={}), \
             patch.object(grade_eval, 'fetch_capability_catalog', return_value=catalog or {'data': [
                 {'id': jm, 'supported_parameters': ['reasoning', 'reasoning_effort'], 'reasoning': {'supported_efforts': ['high', 'low']}}
                 for jm in grade_eval.JUDGE_PANEL]}):
            grade_eval.use_gateway('cometapi', models_file)
        return asked.get('models')


class JudgeGateway(Panel):
    def test_cometapi_gives_judges_more_attempts_and_their_own_efforts(self):
        catalog = {'data': [{'id': jm, 'supported_parameters': ['reasoning', 'reasoning_effort'],
                             'reasoning': {'supported_efforts': ['high', 'low']}} for jm in grade_eval.JUDGE_PANEL[:2]]
                   + [{'id': grade_eval.JUDGE_PANEL[2], 'supported_parameters': ['reasoning'], 'reasoning': {'mandatory': True}}]}
        self.use_cometapi_panel(self.registry(), catalog)
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


ITEM = {'item_id': 'i', 'question': 'q', 'gold_answer': 'a', 'response': 'b', 'answer_type': 'string',
        'answerable': True, 'eval_mode': 'grounded_answerable', 'numeric_precision_rule': None}


def reply(model_id, prompt_tokens=50):
    return {'content': '{"grade": "CORRECT"}', 'model_requested': model_id, 'cost_usd': 0.0, 'prompt_tokens': prompt_tokens}


class JudgeRoutes(Panel):
    """2026-10-28, on CometAPI: gemini-4-argon read a ~100-token hidden prompt on ~86 % of judge calls and gpt-6.1-sol a
    22,416-token agent prompt on ~6 % (cross-judge prompt-token audit). Owner: those two judge through OpenRouter,
    mimo-v2.6-pro (clean) stays on CometAPI, and each graded file says which judge went where and why."""

    REASON = 'hidden prompt measured on CometAPI, 2026-10-28'

    def test_a_judge_whose_cometapi_route_was_withdrawn_judges_through_openrouter_and_says_why(self):
        argon, sol, mimo = grade_eval.JUDGE_PANEL
        asked = self.use_cometapi_panel(self.registry(**{argon: {'cometapi_excluded': self.REASON},
                                                         sol: {'cometapi_excluded': self.REASON}}))
        self.assertEqual(asked, [mimo])
        self.assertEqual(grade_eval.JUDGE_GATEWAY, {argon: 'openrouter', sol: 'openrouter', mimo: 'cometapi'})
        self.assertEqual(grade_eval.JUDGE_ROUTE_NOTES, {argon: self.REASON, sol: self.REASON})

    def test_a_judge_with_no_route_and_no_reason_fails_loud(self):
        with self.assertRaises(RuntimeError):
            self.use_cometapi_panel(self.registry(**{grade_eval.JUDGE_PANEL[0]: {}}))

    def test_each_judge_call_goes_through_its_own_gateway(self):
        argon, sol, mimo = grade_eval.JUDGE_PANEL
        grade_eval.JUDGE_GATEWAY.update({argon: 'openrouter', sol: 'openrouter', mimo: 'cometapi'})
        went = {}
        def openrouter(**kw):
            went[kw['model_id']] = 'openrouter'
            return reply(kw['model_id'])
        def cometapi(model_id, *args, **kw):
            went[model_id] = 'cometapi'
            return reply(model_id)
        with patch.object(grade_eval, 'call_llm', side_effect=openrouter), \
             patch.object(grade_eval, 'call_cometapi', side_effect=cometapi):
            graded, _ = grade_eval.judge_qa_with_panel(dict(ITEM))
        self.assertEqual(went, {argon: 'openrouter', sol: 'openrouter', mimo: 'cometapi'})
        self.assertEqual(set(graded['judge_votes']), set(grade_eval.JUDGE_PANEL))

    def test_a_cometapi_judge_reply_that_read_more_tokens_than_its_prompt_has_characters_is_resent(self):
        grade_eval.JUDGE_GATEWAY.update({jm: 'cometapi' for jm in grade_eval.JUDGE_PANEL})
        grade_eval.JUDGE_RETRIES = 3
        sizes = {jm: iter([22_476, 900]) for jm in grade_eval.JUDGE_PANEL}
        def cometapi(model_id, *args, **kw):
            return reply(model_id, next(sizes[model_id]))
        with patch.object(grade_eval, 'call_cometapi', side_effect=cometapi):
            graded, telemetry = grade_eval.judge_qa_with_panel(dict(ITEM))
        self.assertEqual({t['prompt_tokens'] for t in telemetry}, {900})

        sizes = {jm: iter([22_476] * 5) for jm in grade_eval.JUDGE_PANEL}
        with patch.object(grade_eval, 'call_cometapi', side_effect=cometapi), self.assertRaises(RuntimeError):
            grade_eval.judge_qa_with_panel(dict(ITEM))


if __name__ == '__main__':
    unittest.main()
