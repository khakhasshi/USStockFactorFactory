from dataclasses import asdict
from datetime import date, datetime, timezone
from unittest.mock import patch
import pytest
import polars as pl

from backend.app.trade_plans import (build_plan, schedule, calendar, engine_config,
    ManualReplay, digest, CreateReq, FillReq)
from backend.app.backtest.engine import EventBacktestConfig, StepEventBacktester, _weighted_sleeve_specs
from backend.app.backtest.risk import ExitPolicyConfig


def frozen(market="us",mode="long_only",**kwargs):
    cfg=EventBacktestConfig(market=market,mode=mode,universe_n=100,top_fraction=.25,
        long_gross_target=.9,short_gross_target=.9 if mode=="long_short" else 0,
        max_gross_leverage=2 if mode=="long_short" else 1,**kwargs)
    return {"config":asdict(cfg),"capital":100000,"first_execution":"2026-09-21",
        "sleeves":_weighted_sleeve_specs([{"expression":"rank(close)","weight":1}]),"source_hash":"frozen"}


def rows(day):
    return [{"trade_date":day,"ts_code":f"S{i}","name":f"S{i}","univ_rank":i+1,
        "factor":float(i),"raw_open":10.,"raw_close":10.,"raw_high":10.2,"raw_low":9.8,
        "vol":1000000.,"amount":10000000.,"adjustment_factor":1.,"can_buy_open_proxy":True,
        "can_sell_open_proxy":True,"_adv20_prev":1000000.,"_atr_pct":.02,"_vol20_prev":.2} for i in range(4)]


def loader(**kwargs):
    cal=calendar(kwargs["market"],date.fromisoformat(kwargs["start"]),date.fromisoformat(kwargs["end"]))
    days=cal.sessions_in_range(kwargs["start"],kwargs["end"])
    return pl.DataFrame([r for d in days for r in rows(d.date())]),None


NOW=datetime(2026,9,20,10,tzinfo=timezone.utc)


@pytest.mark.parametrize("market,mode",[("us","long_only"),("us","long_short"),("ashare","long_only")])
def test_targets_equal_event_engine_and_no_synthetic_fills(market,mode):
    f=frozen(market,mode)
    plan=build_plan(f,[],date(2026,9,21),now=NOW,loader=loader)
    spec=f["sleeves"][0]; engine=StepEventBacktester(engine_config(f,spec))
    expected=engine._target_quantities(rows(date(2026,9,18)),100000)
    actual={i["symbol"]:i["target_quantity"] for i in plan["sleeves"][0]["items"]}
    assert actual==expected
    assert plan["cash"]==100000 and plan["nlv"]==100000
    assert all(i["current_quantity"]==0 for i in plan["sleeves"][0]["items"])
    assert plan["signal_date"]=="2026-09-18"


def test_cadence_holidays_and_invalid_first_day():
    _,signal,due=schedule("us",date(2026,9,4),date(2026,9,14),5)
    assert signal==date(2026,9,11) and due # Labor Day excluded
    with pytest.raises(ValueError):schedule("us",date(2026,9,5),date(2026,9,8),5)
    cal,signal,due=schedule("ashare",date(2026,9,30),date(2026,10,9),5)
    assert signal==date(2026,10,8) and not due


def test_expiry_and_incomplete_signal_fail_closed():
    with pytest.raises(ValueError,match="EXPIRED"):
        build_plan(frozen(),[],date(2026,9,21),now=datetime(2026,9,21,14,tzinfo=timezone.utc),loader=loader)
    with pytest.raises(ValueError,match="DATA_NOT_READY"):
        build_plan(frozen(),[],date(2026,9,21),now=datetime(2026,9,18,19,tzinfo=timezone.utc),loader=loader)
    def empty(**kwargs):return pl.DataFrame(rows(date(2026,9,17))),None
    with pytest.raises(ValueError,match="DATA_GAP"):
        build_plan(frozen(),[],date(2026,9,21),now=NOW,loader=empty)


def test_partial_manual_fill_is_actual_position_next_day():
    entry={"sleeve_id":"F01","kind":"fill","symbol":"S3","quantity":100,"price":10.,
        "fees":1.,"date":"2026-09-21","atr_pct":.02,"sequence":1}
    p=build_plan(frozen(),[entry],date(2026,9,22),now=datetime(2026,9,21,21,tzinfo=timezone.utc),loader=loader)
    assert p["cash"]==98999
    assert p["nlv"]==99999
    item=p["sleeves"][0]["items"][0]
    assert item["current_quantity"]==100 and item["target_quantity"]==100
    assert item["delta_quantity"]==0 and not p["rebalance"]


def test_sleeves_are_not_netted():
    f=frozen();f["sleeves"]=_weighted_sleeve_specs([{"expression":"rank(close)","weight":3},{"expression":"rank(close)","weight":1}])
    p=build_plan(f,[],date(2026,9,21),now=NOW,loader=loader)
    assert len(p["sleeves"])==2
    assert [s["nlv"] for s in p["sleeves"]]==[75000,25000]


def test_ashare_t_plus_one_and_cash_checks():
    cfg=engine_config(frozen("ashare"),frozen("ashare")["sleeves"][0])
    entry={"kind":"fill","symbol":"S3","quantity":100,"price":10.,"fees":1.,"date":"2026-09-21","atr_pct":.02}
    runner=ManualReplay(cfg,[entry,{**entry,"quantity":-100}])
    with pytest.raises(ValueError,match="T\\+1"):
        runner.step(trade_date=date(2026,9,21),rows=rows(date(2026,9,21)),next_trade_date=date(2026,9,22),rebalance=False)


def test_corporate_actions_never_forge_actual_shares():
    runner=ManualReplay(EventBacktestConfig(),[])
    runner.positions={"S0":100};runner.last_adjustment={"S0":1}
    with pytest.raises(ValueError,match="CORPORATE_ACTION_REVIEW"):
        runner._apply_corporate_actions(date(2026,9,21),{"S0":{"adjustment_factor":2}})
    assert runner.positions["S0"]==100


def test_nonfinite_request_rejected():
    with pytest.raises(ValueError):CreateReq(backtest_id=1,name="a",first_execution="2026-09-21",capital=float("nan"))
    with pytest.raises(ValueError):FillReq(external_id="x",sleeve_id="F01",date="2026-09-21",price=float("inf"))


def test_new_position_has_provisional_risk_levels():
    f=frozen(exit_policy=ExitPolicyConfig(fixed_stop_loss_pct=.08,fixed_take_profit_pct=.2))
    p=build_plan(f,[],date(2026,9,21),now=NOW,loader=loader)
    item=p['sleeves'][0]['items'][0]
    assert item['risk_levels_provisional']
    assert item['stop_price']==pytest.approx(9.2) and item['take_profit_price']==12


def test_code_change_fail_closed():
    f=frozen();f['decision_code_hash']='different'
    with pytest.raises(ValueError,match='CODE_CHANGED'):
        build_plan(f,[],date(2026,9,21),now=NOW,loader=loader)


def test_risk_exit_never_creates_confirmed_sale():
    f=frozen(exit_policy=ExitPolicyConfig(fixed_stop_loss_pct=.05))
    cfg=engine_config(f,f['sleeves'][0])
    entry={"kind":"fill","symbol":"S3","quantity":100,"price":10.,"fees":1.,"date":"2026-09-21","atr_pct":.02}
    r=ManualReplay(cfg,[entry])
    r.step(trade_date=date(2026,9,21),rows=rows(date(2026,9,21)),next_trade_date=date(2026,9,22),rebalance=False)
    falling=rows(date(2026,9,22))
    for row in falling:row.update(raw_open=9.,raw_high=9.2,raw_low=8.8,raw_close=9.)
    r.step(trade_date=date(2026,9,22),rows=falling,next_trade_date=date(2026,9,23),rebalance=False)
    assert r.positions['S3']==100
    assert r.cash==98999
    assert any(o['target_quantity']==0 for o in r.pending_orders)
    assert r.risk_notices


def test_delayed_reconciliation_does_not_enable_historical_new_plan():
    when=datetime(2026,9,25,22,tzinfo=timezone.utc)
    with pytest.raises(ValueError,match='EXPIRED'):
        build_plan(frozen(),[],date(2026,9,22),now=when,loader=loader)
    p=build_plan(frozen(),[],date(2026,9,22),now=when,loader=loader,reconciliation_only=True)
    assert p['nlv']==100000
