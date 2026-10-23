"""Offline paired article-cluster bootstrap; never calls providers or edits benchmark data."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, random
from collections import Counter,defaultdict
from pathlib import Path

MODELS=('gpt-6-astra','gpt-6.1-sol')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def identity(row):return (row['article_id'],row['item_id'],row['eval_mode'])
def percentile(values):
    values=sorted(values);n=len(values)
    return [round(values[max(0,int(.025*n))],2),round(values[min(n-1,int(.975*n))],2)]
def run_analysis(run,n_boot,seed,astra_file=None,sol_file=None,prune_file=None,scorer=None):
    if n_boot<2000:raise ValueError('At least 2000 samples required')
    if run is not None:
        index=json.loads((run/'combined-router-evidence.json').read_text())
        refs={r['model_id']:r for r in index['models']}
        if any(x is not None for x in (astra_file,sol_file,prune_file,scorer)):
            raise ValueError('Use either --run-dir or explicit projection paths')
        paths={m:Path(refs[m]['result']['path']) for m in MODELS}
        source=run.parents[1]/'scripts/rollup_and_plot.py'
        frozen=json.loads((run/'grading/router-evidence.json').read_text())['frozen_stage6']
        condition=index['condition']
    else:
        if any(x is None for x in (astra_file,sol_file,prune_file,scorer)):
            raise ValueError('Published projections require --astra-file, --sol-file, --prune-file and --scorer')
        paths={MODELS[0]:astra_file,MODELS[1]:sol_file}
        refs={m:{'result':{'sha256':sha(path)}} for m,path in paths.items()}
        source=scorer
        prune=json.loads(prune_file.read_text())
        frozen=prune.get('frozen_stage6',prune)
        frozen={k:frozen[k] for k in ('stage6_saturated_items_all','stage6_saturated_items_dropped')}
        condition='codex-app-server-isolated-v1'
    loaded={m:json.loads(path.read_text()) for m,path in paths.items()}
    expected={loaded[m]['grading_source_sha256']['rollup_and_plot.py'] for m in MODELS}
    if expected!={sha(source)}:raise ValueError('Exact scoring source differs')
    spec=importlib.util.spec_from_file_location('frozen_pb_rollup',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    compute=module.compute_metrics_from_subsets
    dropped=set(frozen['stage6_saturated_items_dropped'])
    frozen_sha=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()
    data={};inputs={}
    for model in MODELS:
        path=paths[model]
        if sha(path)!=refs[model]['result']['sha256']:raise ValueError('Result byte hash differs')
        d=loaded[model]
        if d['status']!='graded-complete' or d['condition']!=condition or d['effort']!='low':raise ValueError('Unexpected scored condition')
        if d['model_meta']['model_id']!=model:raise ValueError('Projection model identity differs')
        if d['frozen_stage6_sha256']!=frozen_sha:raise ValueError('Frozen Stage6 source differs')
        q=[r for r in d['qa_responses'] if not(r['eval_mode']=='grounded_answerable' and r['item_id'] in dropped)]
        s=d['summary_responses']
        if compute(q,s)!=d['metrics']:raise ValueError('Stored points do not reproduce')
        if len({identity(r) for r in q+s})!=len(q+s):raise ValueError('Duplicate model row')
        data[model]=(q,s,d)
        inputs[model]={'result_file':str(path),'sha256':sha(path)}
    a_q,a_s,a=data[MODELS[0]];b_q,b_s,b=data[MODELS[1]]
    for left,right in [(a_q,b_q),(a_s,b_s)]:
        l={identity(r):r for r in left};r={identity(x):x for x in right}
        if l.keys()!=r.keys():raise ValueError('Models do not have matched scored items')
        for key,row in l.items():
            for field in ('question','gold_answer','answer_type','prompt_sha256'):
                if row.get(field)!=r[key].get(field):raise ValueError('Matched prompt/task differs')
    article_sets=[{r['article_id'] for r in rows} for rows in (a_q,a_s,b_q,b_s)]
    if any(s!=article_sets[0] for s in article_sets):raise ValueError('Article sets differ')
    articles=sorted(article_sets[0]);by={}
    for model,(q,s,_) in data.items():
        grouped_q=defaultdict(list);grouped_s=defaultdict(list)
        for row in q:grouped_q[row['article_id']].append(row)
        for row in s:grouped_s[row['article_id']].append(row)
        by[model]=(grouped_q,grouped_s)
    samples=defaultdict(list);rng=random.Random(seed)
    for _ in range(n_boot):
        drawn=[rng.choice(articles) for _ in articles];points={}
        for model in MODELS:
            q,s=by[model]
            points[model]=compute([row for art in drawn for row in q[art]],[row for art in drawn for row in s[art]])
        for metric,value in points[MODELS[0]].items():samples[metric].append(value-points[MODELS[1]][metric])
    stats={metric:{'astra':value,'sol_6_1':b['metrics'][metric],
        'astra_minus_sol':round(value-b['metrics'][metric],2),'paired_ci95':percentile(samples[metric])}
        for metric,value in a['metrics'].items()}
    counts={}
    for model,(q,s,_) in data.items():
        counts[model]={'qa':{mode:dict(Counter(row['grading']['final_grade'] for row in q if row['eval_mode']==mode))
            for mode in ('grounded_answerable','grounded_nearmiss','closed_book')},
            'summary':{key:sum(row['grading'][key] for row in s) for key in ('n_claims','n_unsupported','n_contradicted','n_key_facts','n_key_facts_covered')}}
    b_rows={identity(row):row for row in b_q}
    transitions=Counter((row['eval_mode'],row['grading']['final_grade'],b_rows[identity(row)]['grading']['final_grade'])
        for row in a_q if row['grading']['final_grade']!=b_rows[identity(row)]['grading']['final_grade'])
    transition_rows=[{'mode':mode,'astra':ag,'sol_6_1':sg,'count':count} for (mode,ag,sg),count in sorted(transitions.items())]
    return {'classification_changes':transition_rows,'comparison':'Astra minus Sol 6.1','condition':condition,'effort':'low','n_articles':len(articles),
        'matched_qa':len(a_q),'matched_summaries':len(a_s),'bootstrap_samples':n_boot,'seed':seed,
        'resampling':'Same 10 article draws for both models, with replacement; all rows in each selected article retained; frozen prune applied before sampling',
        'metric_implementation':'Exact imported compute_metrics_from_subsets; canonical two-decimal metric rounding and percentile index convention',
        'scoring_source':{'file':str(source),'sha256':sha(source)},'inputs':inputs,
        'frozen_stage6_sha256':a['frozen_stage6_sha256'],'frozen_dropped_count':len(dropped),
        'metrics':stats,'counts':counts,'sol_6_1_high':'Not evaluated or changed by this analysis'}
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,help='Optional private original run; no default')
    parser.add_argument('--astra-file',type=Path);parser.add_argument('--sol-file',type=Path)
    parser.add_argument('--prune-file',type=Path,help='Frozen Stage6 dict or public summary containing frozen_stage6')
    parser.add_argument('--scorer',type=Path,help='Exact rollup_and_plot.py referenced by result source hashes')
    parser.add_argument('--samples',type=int,default=10000);parser.add_argument('--seed',type=int,default=20261023)
    parser.add_argument('--output',type=Path,default=Path('paired-astra-sol.json'))
    args=parser.parse_args()
    if args.run_dir is None and any(x is None for x in (args.astra_file,args.sol_file,args.prune_file,args.scorer)):
        parser.error('Supply --run-dir or all four explicit public projection paths')
    out=run_analysis(args.run_dir,args.samples,args.seed,args.astra_file,args.sol_file,args.prune_file,args.scorer)
    args.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'samples':out['bootstrap_samples'],'articles':out['n_articles'],'metrics':out['metrics'],'counts':out['counts']},indent=2))
