"""经典因子种子库: 首次启动时自动评估并入库 (幂等)."""

import asyncio

from sqlalchemy import func, select

from .db import SessionLocal
from .eval.harness import evaluate
from .factors.similarity import expression_fingerprint
from .models import Factor

CLASSICS = [
    ("mom_120d", "rank(close / delay(close, 120))",
     "中期动量: 过去约半年涨幅高的股票倾向延续强势 (Jegadeesh & Titman)"),
    ("strev_20d", "-rank(ts_delta(close, 20))",
     "短期反转: 过去一个月超涨的股票倾向回落 (Jegadeesh 1990)"),
    ("lowvol_20d", "-rank(ts_std(close / delay(close, 1), 20))",
     "低波动异象: 低波动股票风险调整后收益更高 (Ang et al.)"),
    ("size_amt", "-rank(ts_mean(log(amount), 20))",
     "规模代理: 成交额小的股票存在流动性溢价 (Banz 小市值效应的量额代理)"),
    ("pv_corr_10d", "-rank(ts_corr(rank(vol), rank(close), 10))",
     "量价背离: 价升量缩/价跌量增蕴含反转信息 (Alpha101 风格)"),
    ("range_pos_20d", "rank((close - ts_min(low, 20)) / (ts_max(high, 20) - ts_min(low, 20) + 0.001))",
     "通道位置: 价格处于近月高低区间上沿代表突破强势"),
    ("ma_gap_60d", "rank(close / ts_mean(close, 60))",
     "均线乖离: 价格相对 60 日均线的偏离度量趋势强度"),
    ("vol_spike", "-rank(vol / (ts_mean(vol, 20) + 1))",
     "放量反指: 相对自身放量往往伴随情绪过热后的回落"),
    ("intraday_rev_5d", "-rank(ts_mean((close - open) / (open + 0.001), 5))",
     "日内动量反转: 近 5 日日内涨幅高的股票隔日倾向走弱"),
    ("high_52w", "rank(close / ts_max(high, 250))",
     "52 周新高: 接近年内高点的股票存在锚定效应下的持续动量 (George & Hwang)"),
]


async def seed_classics() -> None:
    async with SessionLocal() as s:
        n = await s.scalar(select(func.count(Factor.id)))
        if n:
            return
    for i, (name, expr, hypo) in enumerate(CLASSICS, 1):
        try:
            m = await asyncio.to_thread(evaluate, expr, 500, 5)
        except Exception:  # noqa: BLE001
            continue
        async with SessionLocal() as s:
            exists = await s.scalar(select(Factor).where(Factor.expression == expr))
            if exists:
                continue
            s.add(Factor(
                name=f"CL{i:03d}_{name}", expression=expr, hypothesis=hypo,
                status="library-admitted", task_name="classic",
                public_metrics=m["public"], gate_metrics=m["gate"],
                fingerprint={"source": "classic-seed", **expression_fingerprint(expr)},
            ))
            await s.commit()
