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

    def test_an_unpriced_token_kind_is_billed_at_input_and_says_so(self):
        pricing = {'currency': 'USD / M Tokens', 'billing_mode': 'tiered_expr', 'ratio': 0.8,
                   'billing_expr': 'len <= 100 ? tier("s", p * 1 + c * 2) : tier("l", p * 1 + c * 2)'}
        usage = {'prompt_tokens': 50, 'completion_tokens': 1, 'cached_tokens': 10, 'cache_write_tokens': 0}
        cost, basis = cc.cost_usd(pricing, usage)
        self.assertAlmostEqual(cost, 0.8 * (50 * 1 + 1 * 2) / 1e6, places=12)
        self.assertEqual(basis['assumptions'], ['cache-read-billed-at-input-rate'])


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


class Retries(unittest.TestCase):
    """A call another channel served is retried (CometAPI spreads calls over channels, one injects its own prompt);
    a refused request is not."""
    def setUp(self):
        import llm_client
        self.llm = llm_client
        llm_client.use_cometapi({'x/m': {'model': 'm', 'wire': 'openai-chat', 'pricing': {}}})
        self.addCleanup(lambda: llm_client._GATEWAY.update(name='openrouter', routes={}))
        from unittest.mock import patch
        self.sleep = patch.object(llm_client.time, 'sleep'); self.sleep.start(); self.addCleanup(self.sleep.stop)

    def test_a_wrong_channel_is_retried(self):
        from unittest.mock import patch
        outcomes = [cc.CometApiError("x/m: CometAPI says None served the call, not 'm'"), {'content': 'ok'}]
        with patch.object(cc, 'call', side_effect=outcomes) as call:
            self.assertEqual(self.llm.call_llm('x/m', [{'role': 'user', 'content': 'q'}], max_retries=3)['content'], 'ok')
        self.assertEqual(call.call_count, 2)

    def test_a_channel_content_filter_is_retried(self):
        # Live 2026-10-28: gpt-6.1-sol's Azure channel refused a judge prompt as invalid_prompt; others may not.
        from unittest.mock import patch
        refusal = cc.CometApiError('HTTP 400 for x/m via CometAPI m: {"error":{"code":"invalid_prompt","message":"flagged"}}')
        with patch.object(cc, 'call', side_effect=[refusal, {'content': 'ok'}]) as call:
            self.assertEqual(self.llm.call_llm('x/m', [{'role': 'user', 'content': 'q'}], max_retries=3)['content'], 'ok')
        self.assertEqual(call.call_count, 2)

    def test_a_refused_request_is_not_retried(self):
        from unittest.mock import patch
        with patch.object(cc, 'call', side_effect=cc.CometApiError('HTTP 400 for x/m via CometAPI m: bad')) as call:
            with self.assertRaises(RuntimeError):
                self.llm.call_llm('x/m', [{'role': 'user', 'content': 'q'}], max_retries=3)
        self.assertEqual(call.call_count, 1)


class PriceExpressions(unittest.TestCase):
    """Every billing_expr shape a routed model used on 2026-10-28, and refusal of anything else before a call is sent."""
    USAGE = {'prompt_tokens': 1_000_000, 'completion_tokens': 1_000_000, 'cached_tokens': 0, 'cache_write_tokens': 0}

    def price(self, expr):
        return {'currency': 'USD / M Tokens', 'billing_mode': 'tiered_expr', 'billing_expr': expr, 'ratio': 0.8}

    def test_single_tier(self):
        # claude-opus-5.5's live expression: one tier, no length condition.
        cost, basis = cc.cost_usd(self.price('tier("base", p * 4 + c * 20 + cr * 0.2 + cc * 5 + cc1h * 8)'), self.USAGE)
        self.assertAlmostEqual(cost, 0.8 * 24, places=6)
        self.assertEqual(basis['tier'], 'base')

    def test_time_of_day_multipliers_are_read_at_the_call_time(self):
        from datetime import datetime, timezone
        # deepseek-v4.1-flash's live expression: x2 on weekdays (1-5, Sunday = 0) at 01-04 and 06-10 UTC.
        expr = ('(tier("base", p * 0.15 + c * 0.6 + cr * 0.003)) * (weekday("UTC") >= 1 && weekday("UTC") < 6 && '
                'hour("UTC") >= 1 && hour("UTC") < 4 ? 2 : 1) * (weekday("UTC") >= 1 && weekday("UTC") < 6 && '
                'hour("UTC") >= 6 && hour("UTC") < 10 ? 2 : 1)')
        base = 0.8 * 0.75
        for when, factor in ((datetime(2026, 10, 27, 2, 0, tzinfo=timezone.utc), 2),   # Tuesday 02:00
                             (datetime(2026, 10, 27, 5, 0, tzinfo=timezone.utc), 1),   # Tuesday 05:00
                             (datetime(2026, 10, 27, 7, 30, tzinfo=timezone.utc), 2),  # Tuesday 07:30
                             (datetime(2026, 10, 31, 2, 0, tzinfo=timezone.utc), 1)):  # Saturday 02:00
            with self.subTest(when=when.isoformat()):
                cost, basis = cc.cost_usd(self.price(expr), self.USAGE, at=when)
                self.assertAlmostEqual(cost, base * factor, places=6)
                self.assertEqual(basis['billed_at'], when.isoformat())

    def test_an_unknown_function_is_refused_when_the_route_is_checked(self):
        catalog = {'models': [{'id': 'm1', 'supported_endpoint_types': ['openai']}],
                   'pricing': [{'id': 'm1', 'pricing': self.price('tier("base", p * 1 + c * 2) * month("UTC")')}]}
        with self.assertRaises(cc.CometApiError):
            cc.route_evidence(catalog, {'model': 'm1', 'wire': 'openai-chat'})


class InjectedInstructions(unittest.TestCase):
    """A CometAPI channel backed by Codex answers under Codex's ~4.2k-token agent prompt (2/24 gpt-6-astra calls on
    2026-10-28; echoed as `instructions`). PrivacyBench never sends instructions, so such a call is not the benchmark."""
    def test_a_response_carrying_instructions_is_refused(self):
        data = {'model': 'gpt-6-astra', 'status': 'completed', 'instructions': 'You are Codex, an agent based on GPT-6.',
                'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'ok'}]}],
                'usage': {'input_tokens': 4229, 'output_tokens': 5}}
        with self.assertRaisesRegex(cc.CometApiError, 'instructions'):
            cc.parse_response('openai-responses', data)

    def test_a_response_without_instructions_is_read(self):
        data = {'model': 'gpt-6-astra', 'status': 'completed', 'instructions': None,
                'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'ok'}]}],
                'usage': {'input_tokens': 13, 'output_tokens': 5}}
        self.assertEqual(cc.parse_response('openai-responses', data)['content'], 'ok')


class QuotaGate(unittest.TestCase):
    """Owner, 2026-10-28: CometAPI "self reloads, 15 bucks at a time, but it take a little time". A quota 403 pauses
    every caller for minutes, then one probe goes first; it never spends a call's attempts; a stall past the cap stops."""
    QUOTA = 'HTTP 403 for x/m via CometAPI m: {"error":{"message":"pre-consume quota failed, remaining quota: $0.0009"}}'

    def gate(self, **kw):
        clock = {'t': 0.0}
        gate = cc.QuotaGate(first_wait=180, max_wait=720, give_up_after=3600,
                            now=lambda: clock['t'], sleep=lambda s: clock.__setitem__('t', clock['t'] + s), **kw)
        return gate, clock

    def test_a_quota_refusal_waits_minutes_and_retries_without_spending_an_attempt(self):
        from unittest.mock import patch
        import llm_client
        llm_client.use_cometapi({'x/m': {'model': 'm', 'wire': 'openai-chat', 'pricing': {}}})
        self.addCleanup(lambda: llm_client._GATEWAY.update(name='openrouter', routes={}))
        gate, clock = self.gate()
        with patch.object(cc, 'QUOTA_GATE', gate), \
             patch.object(cc, 'call', side_effect=[cc.CometApiError(self.QUOTA), cc.CometApiError(self.QUOTA), {'content': 'ok'}]):
            self.assertEqual(llm_client.call_llm('x/m', [{'role': 'user', 'content': 'q'}], max_retries=1)['content'], 'ok')
        self.assertEqual(clock['t'], 180 + 360)

    def test_a_stall_past_the_cap_stops_and_later_calls_fail_fast(self):
        gate, clock = self.gate()
        with self.assertRaises(cc.QuotaExhausted):
            for _ in range(10):
                gate.before_call()
                gate.on_quota()
        with self.assertRaises(cc.QuotaExhausted):
            gate.before_call()

    def test_success_resets_the_backoff(self):
        gate, clock = self.gate()
        gate.before_call(); gate.on_quota()
        gate.before_call(); gate.on_success()
        start = clock['t']
        gate.before_call(); gate.on_quota(); gate.before_call()
        self.assertEqual(clock['t'] - start, 180)

    def test_quota_text_is_recognised(self):
        self.assertTrue(cc.is_quota_refusal(cc.CometApiError(self.QUOTA)))
        self.assertFalse(cc.is_quota_refusal(cc.CometApiError('HTTP 403 for x/m via CometAPI m: forbidden')))
