"""CometAPI transport: what is sent, how usage and cost are read, and which served ids count as the measured model.

The cost cases are measured bills (fixtures/cometapi_billing.json): the computed cost of each call must equal what the
account was charged, or every pb_inference_cost_per_call row from this gateway is wrong.
"""
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
os.environ.setdefault('COMETAPI_KEY', 'test-key')
import cometapi_client as cc

BILLING = json.loads((Path(__file__).parent / 'fixtures' / 'cometapi_billing.json').read_text())
PROMPT = 'You are answering questions ...\n\nSource Article:\nBODY\n\nQuestion: Who?'
PREFIX = PROMPT[:PROMPT.rindex('Question: ')]


class Cost(unittest.TestCase):
    def test_computed_cost_equals_the_measured_bill(self):
        for case in BILLING['cases']:
            with self.subTest(model=case['model']):
                usage = cc.parse_response(case['wire'], case['response'])['usage']
                cost, basis = cc.cost_usd(case['pricing'], usage)
                # The billing counter reads to 0.0001 cents ($0.000001); a delta of two readings can be one step off.
                self.assertAlmostEqual(cost, case['billed_usd'], delta=1.5e-6)
                self.assertEqual(basis['ratio'], 0.8)

    def test_cached_and_written_tokens_are_not_billed_as_fresh_input(self):
        pricing = {'currency': 'USD / M Tokens', 'input': 10, 'output': 50, 'cached_input': 1, 'cache_write': 12.5, 'ratio': 0.8}
        usage = {'prompt_tokens': 1_000_000, 'completion_tokens': 0, 'cached_tokens': 600_000, 'cache_write_tokens': 400_000}
        self.assertAlmostEqual(cc.cost_usd(pricing, usage)[0], 0.8 * (0.6 * 1 + 0.4 * 12.5), places=6)

    def test_long_context_tier_is_chosen_by_prompt_length(self):
        pricing = {'currency': 'USD / M Tokens', 'billing_mode': 'tiered_expr', 'ratio': 1,
                   'billing_expr': 'len <= 100 ? tier("short_context", p * 1 + c * 2) : tier("long_context", p * 10 + c * 20)'}
        usage = {'prompt_tokens': 101, 'completion_tokens': 0, 'cached_tokens': 0, 'cache_write_tokens': 0}
        self.assertEqual(cc.cost_usd(pricing, usage)[1]['tier'], 'long_context')

    def test_an_unpriced_token_kind_fails_loud(self):
        pricing = {'currency': 'USD / M Tokens', 'billing_mode': 'tiered_expr', 'ratio': 0.8,
                   'billing_expr': 'len <= 100 ? tier("s", p * 1 + c * 2) : tier("l", p * 1 + c * 2)'}
        usage = {'prompt_tokens': 50, 'completion_tokens': 1, 'cached_tokens': 10, 'cache_write_tokens': 0}
        with self.assertRaises(cc.CometApiError):
            cc.cost_usd(pricing, usage)


class Usage(unittest.TestCase):
    def test_anthropic_prompt_is_fresh_plus_written_plus_read(self):
        parsed = cc.parse_response('anthropic-messages', BILLING['anthropic_cached_probe'])
        self.assertEqual(parsed['usage']['prompt_tokens'], 70 + 5221)
        self.assertEqual((parsed['usage']['cached_tokens'], parsed['usage']['reasoning_tokens']), (5221, 402))
        self.assertEqual(parsed['content'], 'answer')

    def test_responses_prompt_already_includes_cached(self):
        parsed = cc.parse_response('openai-responses', BILLING['responses_cached_probe'])
        self.assertEqual((parsed['usage']['prompt_tokens'], parsed['usage']['cached_tokens']), (3052, 3049))
        self.assertEqual(parsed['content'], 'answer')

    def test_gemini_completion_counts_thoughts(self):
        data = {'modelVersion': 'gemini-4-argon', 'candidates': [{'finishReason': 'STOP', 'content': {'parts': [
            {'text': 'thinking...', 'thought': True}, {'text': 'answer'}]}}],
            'usageMetadata': {'promptTokenCount': 100, 'candidatesTokenCount': 5, 'thoughtsTokenCount': 40, 'cachedContentTokenCount': 64}}
        parsed = cc.parse_response('gemini', data)
        self.assertEqual((parsed['content'], parsed['usage']['completion_tokens'], parsed['usage']['reasoning_tokens']), ('answer', 45, 40))


class Requests(unittest.TestCase):
    def test_claude_caches_the_shared_prefix_with_sticky_routing(self):
        path, body, headers, record = cc.build_request('anthropic-messages', 'claude-x', PROMPT, temperature=0.0, max_tokens=10,
                                                       reasoning_effort='high', cache_prefix=PREFIX, cache_key='pb-art')
        blocks = body['messages'][0]['content']
        self.assertEqual(''.join(b['text'] for b in blocks), PROMPT)
        self.assertEqual([('cache_control' in b) for b in blocks], [True, False])
        self.assertEqual(headers['session-id'], 'pb-art')
        self.assertEqual((body['output_config'], body['thinking'], 'temperature' in body), ({'effort': 'high'}, {'type': 'adaptive'}, False))
        self.assertIsNone(record['temperature_sent'])

    def test_responses_never_sends_a_session_and_stores_nothing(self):
        path, body, headers, record = cc.build_request('openai-responses', 'gpt-x', PROMPT, temperature=0.0, max_tokens=10,
                                                       reasoning_effort='high', cache_prefix=PREFIX, cache_key='pb-art')
        self.assertEqual(path, '/v1/responses')
        self.assertNotIn('session-id', headers)
        self.assertEqual((body['store'], body['prompt_cache_key'], body['input']), (False, 'pb-art', PROMPT))

    def test_no_effort_keeps_the_temperature(self):
        _, body, _, record = cc.build_request('openai-chat', 'm', PROMPT, temperature=0.0, max_tokens=10,
                                              reasoning_effort=None, cache_prefix=None, cache_key=None)
        self.assertEqual((body['temperature'], 'reasoning_effort' in body, record['temperature_sent']), (0.0, False, 0.0))

    def test_a_prefix_that_is_not_a_prefix_is_refused(self):
        with self.assertRaises(cc.CometApiError):
            cc.build_request('anthropic-messages', 'm', PROMPT, temperature=0.0, max_tokens=10, reasoning_effort='high',
                             cache_prefix='Something else', cache_key='k')


class Identity(unittest.TestCase):
    ROUTE = {'model': 'gpt-5.6-luna', 'wire': 'openai-responses', 'served_ids': ['gpt-5.6-luna-2026-07-09']}

    def test_requested_and_accepted_ids_are_the_measured_model(self):
        self.assertEqual(cc.served_as(self.ROUTE, 'gpt-5.6-luna', 'openai/gpt-5.6-luna'), 'openai/gpt-5.6-luna')
        self.assertEqual(cc.served_as(self.ROUTE, 'gpt-5.6-luna-2026-07-09', 'openai/gpt-5.6-luna'), 'openai/gpt-5.6-luna')

    def test_any_other_served_id_fails_loud(self):
        for served in ('gpt-5.6-luna-2026-08-01', 'gpt-5.6', None):
            with self.subTest(served=served), self.assertRaises(cc.CometApiError):
                cc.served_as(self.ROUTE, served, 'openai/gpt-5.6-luna')


class Routes(unittest.TestCase):
    CATALOG = {'models': [{'id': 'm1', 'supported_endpoint_types': ['openai']}],
               'pricing': [{'id': 'fam', 'matched_models': {'m1': {'currency': 'USD / M Tokens', 'input': 1, 'output': 2, 'ratio': 0.8}},
                            'endpoints': {'responses': {'path': '/v1/responses'}}}]}

    def test_a_declared_wire_is_accepted_with_its_own_price(self):
        evidence = cc.route_evidence(self.CATALOG, {'model': 'm1', 'wire': 'openai-responses'})
        self.assertEqual(evidence['pricing']['input'], 1)

    def test_every_live_endpoints_shape_is_read(self):
        # Live 2026-10-28: gpt-6.1-terra's `endpoints` is a JSON-encoded string; others are path arrays or dicts.
        shapes = {
            'json-string': '{"chat": {"path": "/v1/chat/completions"}, "responses": {"path": "/v1/responses"}}',
            'path-array': ['/v1/chat/completions', '/v1/responses'],
            'dict-of-paths': {'openai-response': '/v1/responses'},
        }
        for name, endpoints in shapes.items():
            with self.subTest(shape=name):
                catalog = {'models': [{'id': 'm1', 'supported_endpoint_types': ['openai']}],
                           'pricing': [{'id': 'm1', 'pricing': {'currency': 'USD / M Tokens', 'input': 1, 'output': 2, 'ratio': 0.8},
                                        'endpoints': endpoints}]}
                self.assertEqual(cc.route_evidence(catalog, {'model': 'm1', 'wire': 'openai-responses'})['wire'], 'openai-responses')

    def test_an_undeclared_wire_or_unlisted_model_fails_loud(self):
        with self.assertRaises(cc.CometApiError):
            cc.route_evidence(self.CATALOG, {'model': 'm1', 'wire': 'anthropic-messages'})
        with self.assertRaises(cc.CometApiError):
            cc.route_evidence(self.CATALOG, {'model': 'm2', 'wire': 'openai-chat'})


if __name__ == '__main__':
    unittest.main()
