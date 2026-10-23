"""Verify public file integrity and recompute all scores/intervals offline."""
import hashlib, importlib.util, json
from pathlib import Path

root=Path(__file__).resolve().parent
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def require(ok,message):
    if not ok: raise ValueError(message)
summary=json.loads((root/'summary.json').read_text())
scorer=root/'scoring/rollup_and_plot.py'
require(digest(scorer)==summary['scoring_source']['sha256'],'Scorer hash differs')
spec=importlib.util.spec_from_file_location('frozen_pb_scorer',scorer)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
frozen=json.loads((root/'frozen-stage6.json').read_text())
frozen_hash=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()
require(frozen==summary['frozen_stage6'] and frozen_hash==summary['frozen_stage6_sha256'],'Frozen prune differs')
dropped=set(frozen['stage6_saturated_items_dropped'])
calls=scored=metric_count=ci_count=0
models=set()
for entry in summary['models']:
    path=root/entry['file']
    require(digest(path)==entry['sha256'],f'{path.name}: projection hash differs')
    data=json.loads(path.read_text())
    model=data['model_meta']['model_id']
    require(model==entry['model_id'] and model not in models,'Duplicate or wrong model')
    models.add(model)
    require(data['status']=='graded-complete' and data['effort']=='low' and data['condition']==summary['condition'],'Unexpected condition')
    require(data['publication']['original_sha256']==entry['original_sha256'],'Original hash anchor differs')
    require(data['grading_source_sha256']['rollup_and_plot.py']==digest(scorer),'Result scorer hash differs')
    require(data['frozen_stage6_sha256']==frozen_hash,'Result prune hash differs')
    qa=data['qa_responses'];summaries=data['summary_responses']
    require(len(qa)==165 and len(summaries)==10,'Incomplete inference')
    require(len({(r['article_id'],r['item_id'],r['eval_mode']) for r in qa+summaries})==175,'Duplicate items')
    for row in qa+summaries:
        t=row['telemetry']
        require(t['reasoning_effort']=='low' and t['cost_usd'] is None and t['model_returned'] is None,'Inference provenance differs')
        require('funding_context' not in t,'Account funding snapshot present')
    eligible=[q for q in qa if not(q['eval_mode']=='grounded_answerable' and q['item_id'] in dropped)]
    metrics=module.compute_metrics_from_subsets(eligible,summaries)
    intervals={k:list(v) for k,v in module.cluster_bootstrap_cis(eligible,summaries,n_boot=2000,seed=20261015).items()}
    require(metrics==data['metrics']==entry['metrics'],f'{model}: metrics differ')
    require(intervals==data['cis_95']==entry['cis_95'],f'{model}: intervals differ')
    calls+=len(qa)+len(summaries);scored+=len(eligible)+len(summaries)
    metric_count+=len(metrics);ci_count+=len(intervals)
require(len(models)==7 and calls==summary['inference_calls']==1225 and scored==summary['scored_evaluations']==1043,'Run counts differ')
print(json.dumps({'status':'verified','models':len(models),'inference_calls':calls,'scored_evaluations':scored,'metric_points':metric_count,'confidence_intervals':ci_count,'provider_calls':0},indent=2))
