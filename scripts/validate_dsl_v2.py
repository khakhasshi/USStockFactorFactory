#!/usr/bin/env python3
"""Read-only actual-panel evaluation/backtest comparison, no production trials."""
import json
import os
from pathlib import Path
import sys
import time
import urllib.request
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.app.eval.harness import evaluate, _evaluate_full_vector
from backend.app.backtest.engine import run_backtest
from backend.app.dsl.engine import upgrade_correlation, expression_profile, DSL_REVISION, validate
from backend.app.config import get_dsl_fields
from scripts.benchmark_python_optimizations import digest, evidence

r='returns(close,1)'; v='vol/(ts_mean(vol,20)+1e-9)'; c=f'gt({r},0)'
terms=[f'ts_corr_v2({r},{v},20)',f'ts_beta({r},{v},20)',f'ts_residual({r},{v},20)',
 f'ts_robust_zscore({r},20)',f'ts_median({r},20)',f'ts_mad({r},20)',
 f'ts_decay_linear({r},20)',f'ts_ewm({r},20)','ts_drawdown(close,20)',
 f'ts_count({c},20)/20',f'ts_bars_since({c},20)/20',f'ts_streak({c},20)/20',
 f'ts_mean_if({r},{c},20)',f'where({c},{r},-{r})',f'cs_residual({r},returns(open,1))',
 f'group_rank({r},gt({v},1))',f'group_zscore({r},gt({v},1))']
expression='rank('+'+'.join(terms)+')'
out=Path(sys.argv[1]); out.parent.mkdir(parents=True,exist_ok=True)
if out.exists(): raise SystemExit('report exists')
report={'dsl_revision':DSL_REVISION,'expression':expression,'profile':expression_profile(expression),'cases':[]}
def record(row):
 report['cases'].append(row)
 out.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str))
 print(json.dumps({k:v for k,v in row.items() if k not in ('metrics','old','new','config')},ensure_ascii=False,default=str),flush=True)
for market,port,eid in [('us',8765,60),('ashare',8767,58)]:
 for scope in ('training','full_vector','event'):
  results=[]; times=[]
  for flag in ('0','1'):
   os.environ['FF_FACTOR_COMPUTE_CACHE']=flag
   start=time.perf_counter()
   if scope=='event':
    result=run_backtest(expression,universe_n=100,start='2025-01-01',end='2026-09-23',market=market,mode='long_only',
      rebalance_every=5,long_gross_target=.9,execution_backend='python',response_trade_limit=10**9,response_daily_limit=None)
    payload=evidence(result)
   else:
    result=(evaluate if scope=='training' else _evaluate_full_vector)(expression,market=market,portfolio_mode='long_only')
    payload={k:v for k,v in result.items() if k!='runtime'}
   results.append(digest(payload)); times.append(round(time.perf_counter()-start,3))
  record({'market':market,'scope':scope,'exact_cache_parity':len(set(results))==1,'hashes':results,'seconds':times,
    'integrity_pass':result.get('integrity',{}).get('all_pass'), 'fills':result.get('stats',{}).get('fills'),
    'input_snapshot':result.get('input_snapshot_id') or result.get('input_provenance',{}).get('snapshot_id')})
 # Explicit v1/v2 audit of affected task expressions; original records untouched.
 raw=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/api/research-records?experiment_id={eid}&limit=500'))
 rows=raw.get('records',raw.get('items',[]))
 experiments=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/api/experiments'))['experiments']
 config=next(x['research_config'] for x in experiments if x['id']==eid)
 task_map={x['name']:x for x in config['engine_config']['tasks']}
 affected=list(dict.fromkeys((x['expression'],x['task_name']) for x in rows if 'ts_corr(' in x.get('expression','')))
 if not affected: affected=[(f'rank(ts_corr({r},{v},20))',next(iter(task_map)))]
 for old,task_name in affected:
  new=upgrade_correlation(old); metrics=[]
  task=task_map[task_name]
  kwargs=dict(market=market,portfolio_mode=task['mode'],universe_n=task['universe_n'],horizon=task['horizon'],
      direction=task.get('direction',1),direction_policy=task.get('direction_policy','both_train_select'),
      cost_bps=task['cost_bps'],panel_glob=config.get('panel_glob'),evaluation_overrides=config.get('evaluation_config'))
  errors=[validate(expr,get_dsl_fields(market)) for expr in (old,new)]
  if any(errors):
   record({'market':market,'scope':'historical_invalid_expression_retained','task_name':task_name,'old':old,'new':new,'errors':errors})
   continue
  for expr in (old,new):
   value=evaluate(expr,**kwargs)
   metrics.append({k:value.get(k) for k in ('direction','public','gate','discovery','input_snapshot_id')})
  record({'market':market,'scope':'explicit_correlation_migration_training_only','old':old,'new':new,
          'task_name':task_name,'config':kwargs,
          'metrics':metrics,'source':'task_history' if any(x.get('expression')==old for x in rows) else 'control_expression'})
report['passed']=all(x.get('exact_cache_parity',True) and x.get('integrity_pass',True) is not False for x in report['cases'])
out.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str))
raise SystemExit(0 if report['passed'] else 1)
