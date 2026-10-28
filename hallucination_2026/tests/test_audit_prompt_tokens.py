"""The prompt-token audit: a call a gateway channel padded with its own prompt is found and re-queued.

Measured 2026-10-28 on CometAPI: a Codex-backed channel adds ~4,216 tokens (it echoes them, and is refused at call time),
and a gemini-4-argon channel adds ~132 tokens without saying so. The audit is the net under every wire.
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_prompt_tokens as audit


def row(mode, item, article, tokens, response='ok'):
    return {'eval_mode': mode, 'item_id': item, 'article_id': article, 'task_type': 'summary' if mode == 'summary' else 'qa',
            'response': response, 'telemetry': {'prompt_tokens': tokens}}


def result(rows):
    return {'model_meta': {'model_id': 't/m'},
            'qa_responses': [r for r in rows if r['task_type'] == 'qa'],
            'summary_responses': [r for r in rows if r['task_type'] == 'summary']}


CLEAN = ([row('grounded_answerable', f'a{a}_q{q}', f'a{a}', 3000 + 100 * a + q) for a in range(4) for q in range(8)]
         + [row('closed_book', f'a{a}_q{q}', f'a{a}', 90 + q) for a in range(4) for q in range(5)]
         + [row('summary', f'a{a}_summary', f'a{a}', 2950 + 100 * a) for a in range(4)])


class Reference(unittest.TestCase):
    def test_a_constant_offset_is_clean_and_a_jump_is_flagged(self):
        new = [dict(r, telemetry={'prompt_tokens': r['telemetry']['prompt_tokens'] + 7}) for r in CLEAN]
        new[3] = dict(new[3], telemetry={'prompt_tokens': new[3]['telemetry']['prompt_tokens'] + 132})
        flagged = audit.against_reference(result(new), result(CLEAN))
        self.assertEqual([f['key'] for f in flagged], [('grounded_answerable', 'a0_q3')])


class SelfConsistency(unittest.TestCase):
    def test_clean_run_has_no_flags(self):
        self.assertEqual(audit.self_consistency(result(CLEAN)), [])

    def test_padding_is_found_in_every_mode(self):
        for index in (5, 34, 52):  # a grounded call, a closed-book call, a summary
            with self.subTest(mode=CLEAN[index]['eval_mode']):
                padded = list(CLEAN)
                padded[index] = dict(CLEAN[index], telemetry={'prompt_tokens': CLEAN[index]['telemetry']['prompt_tokens'] + 132})
                flagged = audit.self_consistency(result(padded))
                self.assertEqual([f['key'] for f in flagged], [(CLEAN[index]['eval_mode'], CLEAN[index]['item_id'])])


class Requeue(unittest.TestCase):
    def test_flagged_rows_become_errors_a_resume_reruns(self):
        data = result(CLEAN)
        audit.requeue(data, [{'key': ('grounded_answerable', 'a1_q2'), 'excess': 4216}])
        rows = {(r['eval_mode'], r['item_id']): r for r in data['qa_responses']}
        self.assertTrue(rows[('grounded_answerable', 'a1_q2')]['response'].startswith('[ERROR: PromptAudit'))
        self.assertEqual(rows[('grounded_answerable', 'a1_q2')]['telemetry']['cost_usd'], 0.0)
        self.assertEqual(sum(r['response'].startswith('[ERROR') for r in data['qa_responses']), 1)


if __name__ == '__main__':
    unittest.main()
