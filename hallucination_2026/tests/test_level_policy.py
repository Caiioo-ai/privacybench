"""A requested level (`high`) snapped exactly as the Caiioo router snaps it.

The router ranks a turn at its level and every adapter snaps that level onto the levels the model declares
(contextflow `src/shared/reasoning-effort/levels.ts`: `effortLadder`, `snapLevel`). PB-NHS measured at the level the
router would send is only comparable if this port picks the same rung, so the fixture is generated from the TS source
itself (`fixtures/snap_level_ts.json`, every declared list in the live OpenRouter catalogue plus crafted cases).
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from api_effort_policy import EffortPolicyError, effort_ladder, resolve_level_effort, snap_level

FIXTURE = json.loads((Path(__file__).parent / 'fixtures' / 'snap_level_ts.json').read_text())


def row(model_id, reasoning, params=('reasoning', 'reasoning_effort', 'max_tokens')):
    entry = {'id': model_id, 'supported_parameters': list(params)}
    if reasoning is not ...:
        entry['reasoning'] = reasoning
    return entry


def catalog(*rows):
    lists = [r for r in FIXTURE['ladderInputs']]
    filler = [{'id': f'fixture/ladder-{i}', 'supported_parameters': ['reasoning', 'reasoning_effort'],
               'reasoning': {'supported_efforts': levels}} for i, levels in enumerate(lists)]
    return {'data': [*rows, *filler]}


class SnapParity(unittest.TestCase):
    def test_live_ladder_matches_ts(self):
        self.assertEqual(effort_ladder(FIXTURE['ladderInputs']), FIXTURE['ladder'])

    def test_crafted_ladder_matches_ts(self):
        crafted = FIXTURE['craftedLadder']
        self.assertEqual(effort_ladder(crafted['inputs']), crafted['ladder'])

    def test_every_case_matches_ts(self):
        for case in FIXTURE['cases']:
            with self.subTest(case=case):
                self.assertEqual(snap_level(case['requested'], case['levels'], FIXTURE['ladder']), case['snapped'])


class ResolveLevel(unittest.TestCase):
    def test_declared_level_is_sent_as_is(self):
        p = resolve_level_effort('t/a', catalog(row('t/a', {'supported_efforts': ['max', 'high', 'low']})), 'high')
        self.assertEqual((p['state'], p['selected_effort']), ('selected-requested-level', 'high'))

    def test_undeclared_level_snaps_toward_the_lower_on_a_tie(self):
        p = resolve_level_effort('t/a', catalog(row('t/a', {'supported_efforts': ['xhigh', 'medium', 'low']})), 'high')
        self.assertEqual((p['state'], p['selected_effort']), ('selected-snapped-level', 'medium'))

    def test_off_is_never_selected(self):
        with self.assertRaises(EffortPolicyError):
            resolve_level_effort('t/a', catalog(row('t/a', {'supported_efforts': ['none']})), 'high')

    def test_no_effort_selection_sends_no_effort(self):
        p = resolve_level_effort('t/a', catalog(row('t/a', {'mandatory': True}, ('reasoning',))), 'high')
        self.assertEqual((p['state'], p['selected_effort']), ('not-applicable-no-selection', None))

    def test_no_reasoning_object_sends_no_effort(self):
        p = resolve_level_effort('t/a', catalog(row('t/a', ..., ('max_tokens',))), 'high')
        self.assertEqual((p['state'], p['selected_effort']), ('not-applicable-no-selection', None))

    def test_gateway_any_null_is_refused(self):
        with self.assertRaises(EffortPolicyError):
            resolve_level_effort('t/a', catalog(row('t/a', {'supported_efforts': None})), 'high')

    def test_policy_identity_names_the_requested_level(self):
        p = resolve_level_effort('t/a', catalog(row('t/a', {'supported_efforts': ['high']})), 'high')
        self.assertEqual(p['policy']['requested_level'], 'high')
        self.assertEqual(p['policy']['version'], 'requested-level-snapped-api-v1')
        self.assertIn('fingerprint', p)


if __name__ == '__main__':
    unittest.main()
