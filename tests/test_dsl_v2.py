import ast
from datetime import date, timedelta
import random
from types import SimpleNamespace

import numpy as np
import polars as pl
from polars.testing import assert_series_equal
import pytest

from backend.app.dsl.engine import parse, validate, required_history, expression_profile, expression_to_latex, upgrade_correlation
from backend.app.dsl.operators_v2 import NAMES
from backend.app.dsl.grammar_v2 import candidates
from backend.app.config import get_dsl_fields, DEFAULT_MINER_TEMPLATE


def frame():
    return pl.DataFrame([{'ts_code':f'S{s}', 'trade_date':date(2021,1,1)+timedelta(days=d),
        'close':100+s+np.sin(d*.7+s)*3+d*.2, 'vol':1000+s*50+np.cos(d*.2+s)*40,
        'open':100.+s+d*.2, 'high':105.+s+d*.2, 'low':95.+s+d*.2,
        'amount':1e6+s*1000+d*100, 'g':float(s//3)} for s in range(6) for d in range(40)])


def values(expr, df):
    return parse(expr, list(df.columns)).apply(df.lazy()).collect()['factor']


EXPRESSIONS = [
    'ts_corr_v2(close,vol,5)', 'where(gt(close,102),close,vol)',
    'ts_count(gt(close,102),5)', 'ts_mean_if(close,gt(close,102),5)',
    'ts_beta(close,vol,5)', 'ts_residual(close,vol,5)', 'ts_median(close,5)',
    'ts_mad(close,5)', 'ts_robust_zscore(close,5)', 'ts_decay_linear(close,5)',
    'ts_ewm(close,5)', 'ts_drawdown(close,5)', 'ts_bars_since(gt(close,102),5)',
    'ts_streak(gt(close,102),5)', 'cs_residual(close,vol)',
    'group_rank(close,g)', 'group_zscore(close,g)',
]


def test_pearson_and_legacy_version_are_distinct():
    df=frame().with_columns((pl.col('close')*2+3).alias('vol'))
    new=values('ts_corr_v2(close,vol,5)',df).drop_nulls().to_numpy()
    old=values('ts_corr(close,vol,5)',df).drop_nulls().to_numpy()
    np.testing.assert_allclose(new,1,atol=1e-9)
    np.testing.assert_allclose(old,.8,atol=1e-9)
    assert upgrade_correlation('ts_corr(close, vol, 5)')=='ts_corr_v2(close, vol, 5)'
    assert expression_profile('ts_corr(close,vol,5)')['correlation_semantics']=='legacy_v1'
    assert expression_profile('ts_corr_v2(close,vol,5)')['correlation_semantics']=='pearson_v2'
    np.testing.assert_allclose(values('ts_beta(vol,close,5)',df).drop_nulls(),2,atol=1e-9)
    np.testing.assert_allclose(values('ts_residual(vol,close,5)',df).drop_nulls(),0,atol=1e-9)
    df=df.with_columns((pl.col('close')+1e12).alias('close'),(pl.col('vol')+2e12).alias('vol'))
    np.testing.assert_allclose(values('ts_corr_v2(close,vol,5)',df).drop_nulls(),1,atol=1e-7)


@pytest.mark.parametrize('expr',EXPRESSIONS)
def test_prefix_causality_group_order_and_cache(expr,monkeypatch):
    from backend.app import factor_compute_cache as cc
    df=frame(); cutoff=date(2021,1,25)
    full=values(expr,df)
    short=df.filter(pl.col('trade_date')<=cutoff)
    expected=df.with_columns(full).filter(pl.col('trade_date')<=cutoff)['factor']
    assert_series_equal(values(expr,short),expected,check_exact=False,abs_tol=1e-10)
    # Alter every future observation, not just remove it.
    future=df.with_columns(pl.when(pl.col('trade_date')>cutoff).then(999999.)
        .otherwise(pl.col('close')).alias('close'))
    actual=future.with_columns(values(expr,future)).filter(pl.col('trade_date')<=cutoff)['factor']
    assert_series_equal(actual,expected,check_exact=False,abs_tol=1e-10)
    scrambled=df.sample(fraction=1,shuffle=True,seed=13)
    ordered=scrambled.with_columns(values(expr,scrambled)).sort('ts_code','trade_date')['factor']
    assert_series_equal(ordered,full,check_exact=False,abs_tol=1e-10)
    snap=SimpleNamespace(frame=df,market='us',store=object(),generation=1,loaded_identity='v2',
        manifest={'snapshot_id':'v2','immutable_inputs_available':True,'code':{'code_sha256':'test'}})
    monkeypatch.setattr(cc,'_CACHE',cc.ColumnCache(1024*1024))
    monkeypatch.setenv('FF_FACTOR_COMPUTE_CACHE','1')
    for _ in range(2):
        cached=cc.factor_column(df,expr,list(df.columns),'us',snapshot=snap)
        assert_series_equal(cached,full,check_exact=True)
    assert expression_to_latex(expr)


def test_window_oracles_and_empty_conditions():
    df=frame().filter(pl.col('ts_code')=='S0'); x=df['close'].to_numpy(); y=df['vol'].to_numpy()
    w=5
    for i in range(w-1,len(x)):
        a,b=x[i-w+1:i+1],y[i-w+1:i+1]
        assert values('ts_corr_v2(close,vol,5)',df)[i]==pytest.approx(np.corrcoef(a,b)[0,1],abs=1e-8)
        beta=np.cov(a,b,ddof=0)[0,1]/np.var(b)
        assert values('ts_beta(close,vol,5)',df)[i]==pytest.approx(beta,abs=1e-8)
        assert values('ts_residual(close,vol,5)',df)[i]==pytest.approx(a[-1]-a.mean()-beta*(b[-1]-b.mean()),abs=1e-8)
        med=np.median(a); mad=np.median(abs(a-med))
        assert values('ts_mad(close,5)',df)[i]==pytest.approx(mad)
        assert values('ts_robust_zscore(close,5)',df)[i]==pytest.approx((a[-1]-med)/(1.4826*mad))
        assert values('ts_decay_linear(close,5)',df)[i]==pytest.approx(a@np.arange(1,6)/15)
        weights=(1-2/6)**np.arange(4,-1,-1)
        assert values('ts_ewm(close,5)',df)[i]==pytest.approx(a@weights/weights.sum())
    assert values('ts_mean_if(close,0,5)',df).null_count()==df.height
    assert values('ts_bars_since(0,5)',df).null_count()==df.height
    assert values('ts_streak(1,5)',df).drop_nulls().to_list()==[5.]*(df.height-4)
    assert values('ts_corr_v2(close,1,5)',df).null_count()==df.height
    assert values('ts_robust_zscore(1,5)',df).null_count()==df.height


def test_nulls_do_not_become_conditions_or_regression_observations():
    df=frame().filter(pl.col('ts_code')=='S0').with_columns(pl.when(pl.col('trade_date')==date(2021,1,5)).then(None).otherwise(pl.col('close')).alias('close'))
    for expr in ('ts_count(gt(close,102),5)','ts_mean_if(close,gt(close,102),5)','ts_corr_v2(close,vol,5)','ts_mad(close,5)'):
        assert values(expr,df)[4:9].null_count()==5
    assert values('where(gt(close,102),1,0)',df)[4] is None
    assert validate('ts_residual(close,close,5)',['close']) is not None
    assert validate('ts_corr_v2(close,vol,1)',['close','vol']) is not None
    assert validate('group_rank(close,industry)',['close']) is not None


def test_cross_section_regression_and_groups():
    df=frame().with_columns((pl.col('close')*3+7).alias('vol'))
    np.testing.assert_allclose(values('cs_residual(vol,close)',df).drop_nulls(),0,atol=1e-10)
    for day in df['trade_date'].unique():
        d=df.filter(pl.col('trade_date')==day)
        actual=values('group_zscore(close,g)',d)
        for g in (0.,1.):
            idx=[i for i,v in enumerate(d['g']) if v==g]
            x=d['close'].to_numpy()[idx]
            np.testing.assert_allclose(actual.to_numpy()[idx],(x-x.mean())/x.std(ddof=1),atol=1e-10)


@pytest.mark.parametrize('operator', ['ts_decay_linear', 'ts_ewm'])
def test_weighted_rolling_null_warmup_and_internal_gaps(operator):
    df = frame().with_columns(pl.when(pl.col('trade_date') == date(2021,1,15))
        .then(None).otherwise(pl.col('close')).alias('close'))
    result = values(f'{operator}(returns(close,1),5)', df)
    raw = values('returns(close,1)', df).to_numpy()
    weights = np.arange(1,6,dtype=float) if operator == 'ts_decay_linear' else (1-2/6)**np.arange(4,-1,-1)
    weights /= weights.sum()
    expected = np.full(df.height,np.nan)
    for start in range(0,df.height,40):
        for i in range(start+4,start+40):
            window = raw[i-4:i+1]
            if np.isfinite(window).all(): expected[i] = window @ weights
    np.testing.assert_allclose(result.to_numpy(),expected,equal_nan=True,atol=1e-12)


def test_history_is_nested_and_not_silently_capped():
    assert required_history('ts_mean_if(ts_residual(close,vol,60),gt(returns(close,20),0),120)')>=181
    assert required_history('ts_quantile(ts_slope(close,60),120,0.5)')>=181
    assert required_history('ts_mean(ts_mean(ts_mean(ts_mean(ts_mean(close,252),252),252),252),252)')>1000


@pytest.mark.parametrize('operator', ['ts_slope','ts_rsquare','ts_resi'])
def test_existing_regression_nulls_remain_missing_without_kernel_panic(operator):
    df=frame().filter(pl.col('ts_code')=='S0')
    x=values('returns(close,1)',df).to_numpy()
    out=values(f'{operator}(returns(close,1),5)',df)
    assert out[:5].null_count()==5
    for i in range(5,len(x)):
        window=x[i-4:i+1]; t=np.arange(1,6)
        beta=np.cov(t,window,ddof=0)[0,1]/np.var(t)
        if operator=='ts_slope': expected=beta
        elif operator=='ts_resi': expected=window[-1]-window.mean()-beta*(5-t.mean())
        else: expected=np.corrcoef(t,window)[0,1]**2
        assert out[i]==pytest.approx(expected,abs=1e-7)


def test_llm_grammar_mutations_and_ml_pool_expose_extensions():
    from backend.app.miner.agent import _build_system_prompt, random_expression_for_family, mutate_expression
    from backend.app.qlib_joint import JointModelSpec,_feature_rows
    from backend.app.factors.diversity import mechanisms_for_market
    from backend.app.search_pool import propose_search_seed
    for market in ('us','ashare'):
        fields=get_dsl_fields(market)
        prompt=_build_system_prompt(DEFAULT_MINER_TEMPLATE,fields,market=market)
        assert all(name in prompt for name in NAMES)
        pool=[e for f in mechanisms_for_market(market) for e in candidates(f,fields)]
        assert all(validate(e,fields) is None for e in pool)
        names={n.func.id for e in pool for n in ast.walk(ast.parse(e)) if isinstance(n,ast.Call)}
        assert NAMES<=names
        generated=[random_expression_for_family('momentum',fields,random.Random(i)) for i in range(60)]
        assert any('ts_ewm(' in e or 'ts_decay_linear(' in e for e in generated)
        assert any('ts_corr_v2' in f.expression for f in _feature_rows(JointModelSpec(market=market)))
        assert any(f.name.startswith('DSL2_') for f in _feature_rows(JointModelSpec(market=market)))
        for alg in ('grammar_enumerative','map_elites','mcts_puct','evolutionary','tpe_smac','novelty_search','cegis_repair','gbdt_residual_distill','residual_oof_beam'):
            p=propose_search_seed(family='volume_price_interaction',fields=fields,feedback_nodes=[],algorithms=[alg],rng=random.Random(9))
            assert validate(p.expression,fields) is None
            assert p.metadata['dsl_revision'].startswith('factorfactory.dsl/v2')
    assert any('ts_ewm(' in mutate_expression('rank(returns(close,20))',fields=fields,rng=random.Random(i)) for i in range(80))


def test_llm_legacy_spelling_migrates_only_new_proposal_with_audit(monkeypatch):
    import asyncio
    import json
    from backend.app.miner import agent
    original='rank(ts_corr(returns(close,1),vol/(ts_mean(vol,20)+1e-9),20))'
    async def chat(*args,**kwargs):
        return json.dumps({'candidates':[{'request_id':'r1','expression':original,
            'hypothesis':'Volume confirmation','reflection':'Test standard Pearson',
            'mechanism_family':'volume_price_interaction','change_axis':'new_draft',
            'targeted_failures':[],'expected_effect':'Improve consistency'}]})
    async def mark(*args,**kwargs): pass
    monkeypatch.setattr(agent.llm,'chat',chat)
    monkeypatch.setattr(agent.llm,'mark_validation',mark)
    request={'request_id':'r1','task':{'name':'test','market':'us','mode':'long_only',
        'direction':1,'universe_n':500,'horizon':5},'op':'draft',
        'target_family':'volume_price_interaction','feedback_nodes':[],'base_node':None}
    result=asyncio.run(agent.propose_batch(DEFAULT_MINER_TEMPLATE,[request],{'name':'test'},fields=get_dsl_fields('us')))[0]
    assert result[2]=='llm',result
    assert result[0]==upgrade_correlation(original)
    assert result[3]['original_proposal_expression']==original
    assert result[3]['semantic_normalizations']==['new_proposal_ts_corr_upgraded_to_pearson_v2']
