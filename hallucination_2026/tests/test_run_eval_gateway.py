"""The runner on CometAPI at one level: caching marks, warm-then-fan-out order, the spend cap, and the condition."""
import copy
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_eval

MODEL = {'model_id': 'test/native', 'effective_cutoff': '2026-01-01', 'cutoff_source': 'fixture',
         'default_params': {'reasoning_effort': 'max', 'temperature': 0, 'qa_max_tokens': 100, 'summary_max_tokens': 120}}
ARTICLES = [{'id': f'art{n}', 'published_at': '2026-01-02', 'domain': 'law', 'subtopic': 'ruling',
             'text_committed': True, 'norm_text': f'Article {n} says seven.'} for n in (1, 2)]
ITEMS = [{'id': f'art{n}_q{k}', 'article_id': f'art{n}', 'answerable': True, 'domain': 'law', 'subtopic': 'ruling',
          'question': f'Question {k}?', 'gold_answer': 'seven', 'answer_type': 'string', 'numeric_precision_rule': None,
          'generator_model': 'fixture', 'modes': ['grounded_answerable', 'closed_book']} for n in (1, 2) for k in (1, 2, 3)]
CATALOG = {'data': [{'id': 'test/native', 'reasoning': {'supported_efforts': ['max', 'xhigh', 'medium', 'low']},
                     'supported_parameters': ['reasoning', 'reasoning_effort', 'temperature', 'max_tokens']},
                    # Another model's list places `high` on the ladder, as the live catalogue does.
                    {'id': 'test/other', 'reasoning': {'supported_efforts': ['xhigh', 'high', 'medium']},
                     'supported_parameters': ['reasoning', 'reasoning_effort']}]}
ROUTE = {'model': 'native', 'wire': 'anthropic-messages', 'pricing': {'currency': 'USD / M Tokens', 'input': 1, 'output': 2, 'ratio': 0.8}}


class GatewayRunner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.calls, self.lock = [], threading.Lock()
        self.in_flight = {'warm_done': set()}
        for target, value in (('gateway_name', 'cometapi'), ('loaded_route', ROUTE)):
            p = patch.object(run_eval, target, return_value=value)
            p.start(); self.addCleanup(p.stop)
        p = patch.object(run_eval, 'call_llm', side_effect=self.provider)
        p.start(); self.addCleanup(p.stop)

    def provider(self, **kw):
        with self.lock:
            self.calls.append(kw)
        return {'content': 'seven', 'model_requested': kw['model_id'], 'model_returned': kw['model_id'],
                'reasoning_effort': kw['reasoning_effort'], 'temperature': kw['temperature'], 'finish_reason': 'stop',
                'cost_usd': 0.01, 'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3}

    def run_model(self, **kwargs):
        return run_eval.run_model_evaluation(copy.deepcopy(MODEL), copy.deepcopy(ARTICLES), copy.deepcopy(ITEMS),
                                             Path(self.tmp.name), '2099-01-01', max_workers=8,
                                             capability_catalog=copy.deepcopy(CATALOG), effort_level='high', **kwargs)

    def test_level_is_snapped_and_named_in_the_condition(self):
        data = json.loads(self.run_model().read_text())
        self.assertEqual({c['reasoning_effort'] for c in self.calls}, {'medium'})
        self.assertEqual(data['inference_contract']['condition'], 'cometapi-api-level-high-v1')
        self.assertEqual(data['inference_contract']['gateway_route'], ROUTE)

    def test_only_grounded_calls_carry_the_article_prefix_and_key(self):
        self.run_model()
        for c in self.calls:
            prompt = c['messages'][0]['content']
            if 'Source Article:' in prompt:
                self.assertEqual(c['cache_prefix'], prompt[:prompt.rindex('Question: ')])
                self.assertTrue(c['cache_key'].startswith('privacybench-art'))
            else:
                self.assertEqual((c['cache_prefix'], c['cache_key']), (None, None))

    def test_each_articles_first_cached_call_runs_before_its_siblings(self):
        self.run_model()
        keyed = [c['cache_key'] for c in self.calls if c['cache_key']]
        first_two = keyed[:2]
        self.assertEqual(sorted(first_two), ['privacybench-art1', 'privacybench-art2'])
        self.assertEqual(len(keyed), 6)
        # The warm calls are the first two calls of the whole run, before any other task is sent.
        self.assertEqual([c['cache_key'] for c in self.calls[:2]], keyed[:2])

    def test_spend_cap_stops_sending_and_records_errors(self):
        data = json.loads(self.run_model(max_model_cost_usd=0.025).read_text())
        rows = data['qa_responses'] + data['summary_responses']
        capped = [r for r in rows if 'SpendCap' in r['response']]
        self.assertGreater(len(capped), 0)
        self.assertLess(len(self.calls), len(rows))


if __name__ == '__main__':
    unittest.main()
