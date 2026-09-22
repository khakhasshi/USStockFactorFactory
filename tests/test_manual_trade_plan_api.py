"""Real PostgreSQL tests isolated in a disposable, uniquely named schema."""
import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from backend.app import trade_plans as tp
from backend.app.models import Backtest
from tests.test_manual_trade_plans import frozen, loader


def test_postgres_api_freeze_import_idempotency_and_reconciliation():
    asyncio.run(run_api())


async def run_api():
    schema="manual_plan_test_"+uuid.uuid4().hex
    admin=create_async_engine("postgresql+asyncpg://jiangjingzhe@localhost:5432/factor_factory_us_8765")
    async with admin.begin() as c:await c.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine=create_async_engine("postgresql+asyncpg://jiangjingzhe@localhost:5432/factor_factory_us_8765",connect_args={"server_settings":{"search_path":schema}})
    sessions=async_sessionmaker(engine,expire_on_commit=False)
    original_build=tp.build_plan
    def test_build(*args,**kwargs):
        kwargs.update(loader=loader,archive=False,now=datetime(2026,9,21,21,tzinfo=timezone.utc))
        # Initial plan intentionally tests historical fixture generation.
        kwargs["reconciliation_only"]=True
        return original_build(*args,**kwargs)
    try:
        async with engine.begin() as c:
            await c.run_sync(lambda sync:tp.Base.metadata.create_all(sync,tables=[Backtest.__table__,tp.PlanInstance.__table__,tp.PlanBatch.__table__,tp.ManualEntry.__table__]))
        async with sessions() as s:
            f=frozen()
            s.add(Backtest(id=841,experiment_id=1,status="done",error="",params={"factors":[{"expression":"rank(close)","weight":1}]},result={"config":f["config"],"integrity":{"all_pass":True},"protocol":"step_event_v2_weighted_sleeves_v1","curve":{"dates":["2025-01-02","2026-09-18"]}}))
            await s.commit()
        class BeforeCreate(datetime):
            @classmethod
            def now(cls,tz=None):return datetime(2026,9,20,10,tzinfo=timezone.utc)
        with patch.object(tp,"SessionLocal",sessions),patch.object(tp,"SERVICE_MARKET","us"),patch.object(tp,"build_plan",test_build),patch.object(tp,"datetime",BeforeCreate):
            await tp.initialize()
            app=FastAPI();app.include_router(tp.router)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test") as client:
                r=await client.post('/api/trade-plans',json={"name":"test","backtest_id":841,"first_execution":"2026-09-21","capital":100000})
                assert r.status_code==200,r.text
                p=r.json();ident=p['id'];base=f'/api/trade-plans/{ident}'
                r=await client.post(base+'/generate',json={"execution_date":"2026-09-21"})
                assert r.status_code==200,r.text
                b=r.json()
                duplicate=await client.post(base+'/generate',json={"execution_date":"2026-09-21"})
                assert duplicate.json()['id']==b['id']
                blocked=await client.post(base+'/generate',json={"execution_date":"2026-09-22"})
                assert blocked.status_code==409
                # No actual trades invented by generating a plan.
                assert (await client.get(base)).json()['entries']==[]
                assert (await client.get(base+f'/batches/{b["id"]}.csv')).status_code==200
                # Database-level published evidence cannot be overwritten.
                async with sessions() as s:
                    try:
                        await s.execute(text("UPDATE manual_plan_batches SET snapshot='{}'"));await s.commit()
                        raise AssertionError('immutable trigger missing')
                    except Exception as exc:
                        assert 'immutable' in str(exc)
                        await s.rollback()
                class AfterClose(datetime):
                    @classmethod
                    def now(cls,tz=None):return datetime(2026,9,21,21,tzinfo=timezone.utc)
                with patch.object(tp,'datetime',AfterClose):
                    payload={"external_id":"trade-1","sleeve_id":"F01","date":"2026-09-21","symbol":"S3","quantity":100,"price":10.,"fees":1.}
                    a=await client.post(base+'/entries',json=payload)
                    assert a.status_code==200,a.text
                    a=await client.post(base+'/entries',json=payload)
                    assert a.json()['duplicate'] is True
                    assert (await client.post(base+'/entries',json={**payload,"price":11})).status_code==409
                    a=await client.post(base+'/import-csv',json={"csv_text":"external_id,sleeve_id,date,symbol,quantity,price,fees\ntrade-2,F01,2026-09-21,S3,1,10,0\nbad,F01,2026-09-21,MISSING,1,10,0\n"})
                    assert a.json()['saved_or_duplicate']==1 and a.json()['failed']==1,a.text
                    a=await client.post(base+f'/batches/{b["id"]}/reconcile',json={"note":"交割已核对，其他均未成交"})
                    assert a.status_code==200,a.text
                    assert a.json()['cash']==98989
                    assert (await client.post(base+'/entries',json={**payload,"external_id":"too-late"})).status_code==409
                    r=await client.post(base+'/generate',json={"execution_date":"2026-09-22"})
                    assert r.status_code==200,r.text
                    assert r.json()['snapshot']['sleeves'][0]['items'][0]['current_quantity']==101
                assert (await client.post(base+'/status',json={"status":"archived"})).status_code==200
                assert (await client.post(base+'/status',json={"status":"active"})).status_code==409
    finally:
        await engine.dispose()
        async with admin.begin() as c:await c.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()
