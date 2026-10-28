import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_eval

MODEL = {'model_id': 'test/native', 'effective_cutoff': '2026-01-01', 'cutoff_source': 'fixture',
         'default_params': {'reasoning_effort': 'low', 'temperature': 0, 'qa_max_tokens': 100, 'summary_max_tokens': 120}}
ARTICLE = {'id':'article','published_at':'2026-01-02','domain':'law','subtopic':'ruling',
           'text_committed':True,'norm_text':'The source says seven.'}
ITEM = {'id':'q','article_id':'article','answerable':True,'domain':'law','subtopic':'ruling',
        'question':'How many?','gold_answer':'seven','answer_type':'string','numeric_precision_rule':None,
        'generator_model':'fixture','modes':['grounded_answerable','closed_book']}
CATALOG = {'data':[{'id':'test/native','reasoning':{'supported_efforts':['max','xhigh','high','low']},
                     'supported_parameters':['reasoning','reasoning_effort','temperature','max_tokens']}]}


def fake_provider(**kwargs):
    return {'content':'seven','model_requested':kwargs['model_id'],'model_returned':kwargs['model_id'],
            'reasoning_effort':kwargs['reasoning_effort'],'temperature':kwargs['temperature'],
            'finish_reason':'stop','cost_usd':0.001,'total_tokens':3}


class ApiRunnerFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.catalog = patch.object(run_eval, 'fetch_capability_catalog', return_value=copy.deepcopy(CATALOG), create=True)
        self.catalog.start(); self.addCleanup(self.catalog.stop)
        self.provider = patch.object(run_eval, 'call_llm', side_effect=fake_provider)
        self.call = self.provider.start(); self.addCleanup(self.provider.stop)

    def run_model(self, model=None, items=None, **kwargs):
        return run_eval.run_model_evaluation(copy.deepcopy(model or MODEL), [copy.deepcopy(ARTICLE)],
            copy.deepcopy(items or [ITEM]), self.directory, '2099-01-01', max_workers=1, **kwargs)


class ApiRunnerRegressionTests(ApiRunnerFixture):
    def test_highest_native_effort_replaces_stale_low_registry_default(self):
        self.run_model()
        self.assertEqual([c.kwargs['reasoning_effort'] for c in self.call.call_args_list], ['max']*3)

    def test_legacy_cache_is_refused_before_spending_and_preserved(self):
        path=self.directory/'2099-01-01-test-native.json'
        old={'qa_responses':[], 'summary_responses':[], 'grading_summary':{'stale':True}}
        path.write_text(json.dumps(old)); before=path.read_bytes()
        with self.assertRaisesRegex(ValueError, '[Cc]ache|[Rr]esume|legacy'):
            self.run_model()
        self.call.assert_not_called(); self.assertEqual(path.read_bytes(),before)

    def test_changed_prompt_cache_is_refused_before_spending(self):
        path=self.run_model(); before=path.read_bytes(); self.call.reset_mock()
        changed=copy.deepcopy(ITEM); changed['question']='How many really?'
        with self.assertRaisesRegex(ValueError, '[Cc]ache|[Rr]esume'):
            self.run_model(items=[changed])
        self.call.assert_not_called(); self.assertEqual(path.read_bytes(),before)


class CapabilityPolicyTests(unittest.TestCase):
    def resolve(self, reasoning, params=None):
        from api_effort_policy import resolve_effort
        row={'id':'test/native','supported_parameters':params if params is not None else ['reasoning','reasoning_effort'],
             'reasoning':reasoning}
        return resolve_effort('test/native',{'data':[row]})

    def test_declared_order_is_authoritative_and_native_value_is_verbatim(self):
        self.assertEqual(self.resolve({'supported_efforts':['max','xhigh','none']})['selected_effort'],'max')
        self.assertEqual(self.resolve({'supported_efforts':['medium','high']})['selected_effort'],'medium')
        self.assertEqual(self.resolve({'supported_efforts':['high','none']})['selected_effort'],'high')

    def test_gateway_any_absent_malformed_and_unknown_are_distinct(self):
        cases=[({'supported_efforts':None},'gateway-any'),
               ({'supported_efforts':['future','max']},'unknown effort'),
               ({'supported_efforts':'high'},'malformed'),
               ({'supported_efforts':[]},'empty'),
               ({'supported_efforts':['max','max']},'malformed'),
               ({'supported_efforts':['ultra','none']},'unknown effort'),
               ({'supported_efforts':['ultra','high']},'unknown effort'),
               ({'supported_efforts':['high'], 'mandatory':'yes'},'malformed'),
               ({'supported_efforts':['high'], 'default_effort':'max'},'conflicts')]
        for declaration,reason in cases:
            with self.subTest(declaration=declaration), self.assertRaisesRegex(ValueError,reason):
                self.resolve(declaration)
        with self.assertRaisesRegex(ValueError,'conflicting'):
            self.resolve({})
        self.assertEqual(self.resolve({}, ['reasoning'])['state'],'not-applicable-no-selection')
        budget=self.resolve({'supports_max_tokens':True}, ['reasoning'])
        self.assertEqual(budget['state'],'not-applicable-budget-only')
        self.assertIsNone(budget['selected_effort'])
        self.assertEqual(self.resolve(None, ['temperature'])['state'],'not-applicable-no-selection')
        with self.assertRaisesRegex(ValueError,'unknown'):
            self.resolve(None, ['reasoning'])

    def test_missing_model_or_ambiguous_catalogue_fails_closed(self):
        from api_effort_policy import resolve_effort
        for catalog in ({}, {'data':[]},{'data':CATALOG['data']*2}):
            with self.subTest(catalog=catalog),self.assertRaises(ValueError):
                resolve_effort('test/native',catalog)

    def test_catalogue_fetch_is_unauthenticated_get(self):
        from api_effort_policy import fetch_capability_catalog, CATALOG_URL
        from unittest.mock import MagicMock
        response=MagicMock(); response.__enter__.return_value.read.return_value=json.dumps(CATALOG).encode()
        with patch('api_effort_policy.urllib.request.urlopen',return_value=response) as fetch:
            result=fetch_capability_catalog()
        self.assertEqual(fetch.call_args.args,(CATALOG_URL,))
        self.assertEqual(result['data'],CATALOG['data'])
        self.assertEqual(len(result['_retrieval']['response_sha256']),64)


class ExactReplayTests(ApiRunnerFixture):
    def test_exact_replay_preserves_grading_and_all_bytes(self):
        path=self.run_model(); data=json.loads(path.read_text())
        data['qa_responses'][0]['grading']={'final_grade':'CORRECT'}
        data['grading_summary']={'proof':'same inference'}
        path.write_text(json.dumps(data)); before=path.read_bytes(); self.call.reset_mock()
        self.run_model()
        self.call.assert_not_called();self.assertEqual(path.read_bytes(),before)
        self.assertEqual(MODEL['default_params']['reasoning_effort'],'low')
        self.assertEqual(data['model_meta']['default_params']['reasoning_effort'],'max')
        self.assertEqual(data['inference_contract']['requested_model_meta']['default_params']['reasoning_effort'],'low')

    def test_changed_effort_capability_temperature_caps_or_gold_refuse_and_preserve(self):
        path=self.run_model(); before=path.read_bytes()
        changes=[]
        for field,value in [('temperature',0.5),('qa_max_tokens',101),('summary_max_tokens',121)]:
            model=copy.deepcopy(MODEL);model['default_params'][field]=value;changes.append({'model':model})
        item=copy.deepcopy(ITEM);item['gold_answer']='eight';changes.append({'items':[item]})
        for kwargs in changes:
            with self.subTest(change=kwargs):
                self.call.reset_mock()
                with self.assertRaisesRegex(ValueError,'Cache refused'):self.run_model(**kwargs)
                self.call.assert_not_called();self.assertEqual(path.read_bytes(),before)
        modified=copy.deepcopy(CATALOG);modified['data'][0]['reasoning']['supported_efforts']=['high','low']
        with patch.object(run_eval,'fetch_capability_catalog',return_value=modified):
            self.call.reset_mock()
            with self.assertRaisesRegex(ValueError,'Cache refused'):self.run_model()
            self.call.assert_not_called();self.assertEqual(path.read_bytes(),before)

    def test_source_policy_or_cached_settings_mutation_refuses_before_any_call(self):
        path=self.run_model(); baseline=json.loads(path.read_text())
        variants=[]
        bad=copy.deepcopy(baseline);bad['inference_contract']['source_sha256']['llm_client.py']='other';variants.append(bad)
        bad=copy.deepcopy(baseline);bad['inference_contract']['effort_policy']['policy']['version']='other';variants.append(bad)
        bad=copy.deepcopy(baseline);bad['qa_responses'][0]['telemetry']['reasoning_effort']='low';variants.append(bad)
        bad=copy.deepcopy(baseline);bad['qa_responses'][0]['request_sha256']='other';variants.append(bad)
        bad=copy.deepcopy(baseline);bad['qa_responses'].append(bad['qa_responses'][0]);variants.append(bad)
        for data in variants:
            path.write_text(json.dumps(data));before=path.read_bytes();self.call.reset_mock()
            with self.assertRaisesRegex(ValueError,'Cache refused'):self.run_model()
            self.call.assert_not_called();self.assertEqual(path.read_bytes(),before)

    def test_any_regenerated_response_discards_all_prior_grading(self):
        path=self.run_model();data=json.loads(path.read_text());data['grading_summary']={'stale':True}
        for row in data['qa_responses']+data['summary_responses']:row['grading']={'stale':True}
        data['qa_responses'][0]['telemetry']['finish_reason']='length'
        path.write_text(json.dumps(data));self.call.reset_mock();self.run_model()
        self.assertEqual(self.call.call_count,1)
        now=json.loads(path.read_text());self.assertNotIn('grading_summary',now)
        self.assertTrue(all('grading' not in row for row in now['qa_responses']+now['summary_responses']))
        self.assertEqual(self.call.call_args.kwargs['max_retries'],1)

    def test_retrieval_time_and_unrelated_models_do_not_invalidate_exact_cache(self):
        path=self.run_model();before=path.read_bytes();self.call.reset_mock()
        modified=copy.deepcopy(CATALOG);modified['_retrieval']={'retrieved_at':'later','response_sha256':'new'}
        modified['data'].append({'id':'unrelated/model'})
        with patch.object(run_eval,'fetch_capability_catalog',return_value=modified):self.run_model()
        self.call.assert_not_called();self.assertEqual(path.read_bytes(),before)

    def test_dry_run_reports_effective_settings_without_calls_or_output(self):
        plan=self.run_model(dry_run=True)
        self.call.assert_not_called();self.assertEqual(list(self.directory.iterdir()),[])
        self.assertEqual(plan['resolved_params']['reasoning_effort'],'max')
        self.assertEqual(plan['requested_params']['reasoning_effort'],'low')
        self.assertEqual(plan['planned_calls'],3);self.assertEqual(plan['provider_calls'],0)

    def test_unknown_capability_refuses_runner_before_any_provider_call(self):
        bad=copy.deepcopy(CATALOG);bad['data'][0]['reasoning']['supported_efforts']=None
        with patch.object(run_eval,'fetch_capability_catalog',return_value=bad):
            with self.assertRaisesRegex(ValueError,'gateway-any'):self.run_model()
        self.call.assert_not_called();self.assertEqual(list(self.directory.iterdir()),[])
        # Codex's description of ultra is not evidence about this API route.
        bad['data'][0]['reasoning']['supported_efforts']=['ultra','high']
        with patch.object(run_eval,'fetch_capability_catalog',return_value=bad):
            with self.assertRaisesRegex(ValueError,'unknown effort'):self.run_model()
        self.call.assert_not_called();self.assertEqual(list(self.directory.iterdir()),[])

    def test_nonconfigurable_effort_is_reported_and_omitted_without_budget_guess(self):
        fixed=copy.deepcopy(CATALOG);fixed['data'][0]['reasoning']={'supports_max_tokens':True}
        fixed['data'][0]['supported_parameters']=['reasoning','temperature','max_tokens']
        with patch.object(run_eval,'fetch_capability_catalog',return_value=fixed):
            plan=self.run_model(dry_run=True)
            self.assertEqual(plan['effort_policy']['state'],'not-applicable-budget-only')
            self.assertIsNone(plan['resolved_params']['reasoning_effort'])
            self.call.assert_not_called()
            path=self.run_model()
        self.assertTrue(all(c.kwargs['reasoning_effort'] is None for c in self.call.call_args_list))
        self.assertTrue(all('reasoning_max_tokens' not in c.kwargs for c in self.call.call_args_list))
        self.assertEqual(json.loads(path.read_text())['effort_policy']['state'],'not-applicable-budget-only')

    def test_same_slug_different_model_cannot_reuse_output(self):
        path=self.run_model();before=path.read_bytes();self.call.reset_mock()
        model=copy.deepcopy(MODEL);model['model_id']='test:native'
        catalog=copy.deepcopy(CATALOG);catalog['data'][0]['id']='test:native'
        with patch.object(run_eval,'fetch_capability_catalog',return_value=catalog):
            with self.assertRaisesRegex(ValueError,'Cache refused'):self.run_model(model=model)
        self.call.assert_not_called();self.assertEqual(path.read_bytes(),before)

    def test_all_models_cli_validates_every_cache_before_any_spend(self):
        models=[copy.deepcopy(MODEL),dict(copy.deepcopy(MODEL),model_id='test/second')]
        catalog=copy.deepcopy(CATALOG);catalog['data'].append(dict(copy.deepcopy(catalog['data'][0]),id='test/second'))
        for name,payload in [('models',{'models':models}),('articles',{'articles':[ARTICLE]}),('items',{'items':[ITEM]})]:
            (self.directory/f'{name}.json').write_text(json.dumps(payload))
        bad=self.directory/'2099-01-01-test-second.json';bad.write_text('{}');before=bad.read_bytes()
        argv=['run_eval','--all-pilot-models','--models-file',str(self.directory/'models.json'),
              '--articles-file',str(self.directory/'articles.json'),'--items-file',str(self.directory/'items.json'),
              '--results-dir',str(self.directory),'--run-date','2099-01-01']
        with patch.object(sys,'argv',argv),patch.object(run_eval,'fetch_capability_catalog',return_value=catalog):
            with self.assertRaisesRegex(ValueError,'Cache refused'):run_eval.main()
        self.call.assert_not_called();self.assertEqual(bad.read_bytes(),before)

if __name__=='__main__': unittest.main()
