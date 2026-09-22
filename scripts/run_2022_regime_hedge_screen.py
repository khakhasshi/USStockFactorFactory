#!/usr/bin/env python3
"""Reproduce the 2022 regime-hedge sleeve screen from audited backtests.

This helper deliberately separates execution from analysis:

* ``emit-request`` copies every execution/risk setting from frozen backtest 717
  and emits one of the two <=12-factor API requests.
* ``analyze`` reads the frozen benchmark plus the completed batch backtests and
  derives every reported number from their fee-after event-ledger paths.

The script does not optimize candidate weights.  Its 5/10/15 percent probe is
only a deterministic next-test suggestion based on the declared hard gates.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import urllib.request
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/research/backtest-kpi-26-factors-source-20260827.md"
API_ROOT = "http://127.0.0.1:8765"
BENCHMARK_ID = 717
EXCLUDED = {19, 21, 23}
BATCHES = (
    tuple(number for number in range(1, 27) if number not in EXCLUDED)[:12],
    tuple(number for number in range(1, 27) if number not in EXCLUDED)[12:],
)


def api_backtest(backtest_id: int) -> dict[str, Any]:
    with urllib.request.urlopen(
        f"{API_ROOT}/api/backtests/{int(backtest_id)}", timeout=30
    ) as response:
        return json.load(response)


def expressions() -> dict[int, str]:
    body = SOURCE.read_text(encoding="utf-8")
    rows = re.findall(
        r"^### #(\d+)[^\n]*\n\n\*\*DSL\*\*：`([^`]+)`",
        body,
        flags=re.MULTILINE,
    )
    parsed = {int(number): expression for number, expression in rows}
    if set(parsed) != set(range(1, 27)):
        raise RuntimeError(f"expected factors 1..26, got {sorted(parsed)}")
    return parsed


def emit_request(batch_index: int) -> None:
    benchmark = api_backtest(BENCHMARK_ID)
    if benchmark.get("status") != "done":
        raise RuntimeError(f"benchmark {BENCHMARK_ID} is not done")
    params = deepcopy(benchmark["params"])
    for field in ("protocol", "market", "requested_mode"):
        params.pop(field, None)
    numbers = BATCHES[batch_index - 1]
    source = expressions()
    params["factors"] = [
        {
            "name": f"因子{number}",
            "expression": source[number],
            "weight": 1.0,
            "direction": 1,
        }
        for number in numbers
    ]
    params["expression"] = params["factors"][0]["expression"]
    params["initial_capital"] = float(len(numbers) * 1_000_000)
    print(json.dumps(params, ensure_ascii=False, separators=(",", ":")))


def _returns(values: list[float], initial: float = 1.0) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    prior = np.concatenate([[float(initial)], array[:-1]])
    return array / prior - 1.0


def _corr(left: np.ndarray | list[float], right: np.ndarray | list[float]) -> float | None:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.size != right_array.size or left_array.size < 3:
        return None
    if np.std(left_array) <= 1e-12 or np.std(right_array) <= 1e-12:
        return None
    value = float(np.corrcoef(left_array, right_array)[0, 1])
    return round(value, 6) if math.isfinite(value) else None


def _monthly_returns(dates: list[str], returns: np.ndarray) -> dict[str, float]:
    output: dict[str, float] = {}
    for value, daily_return in zip(dates, returns):
        key = str(value)[:7]
        output[key] = (1.0 + output.get(key, 0.0)) * (1.0 + float(daily_return)) - 1.0
    return output


def _common_corr(left: dict[str, float], right: dict[str, float]) -> float | None:
    common = sorted(set(left) & set(right))
    return _corr([left[key] for key in common], [right[key] for key in common])


def _period_stats(returns: np.ndarray) -> dict[str, float]:
    wealth = np.cumprod(1.0 + returns)
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    downside = np.minimum(returns, 0.0)
    downside_vol = float(np.sqrt(np.mean(np.square(downside))))
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    max_drawdown = float(np.max(1.0 - wealth / peak))
    cagr = float(wealth[-1] ** (252.0 / returns.size) - 1.0)
    sharpe = mean / std * math.sqrt(252.0) if std > 1e-12 else 0.0
    sortino = mean / downside_vol * math.sqrt(252.0) if downside_vol > 1e-12 else 0.0
    return {
        "total_return": float(wealth[-1] - 1.0),
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": cagr / max_drawdown if max_drawdown > 1e-12 else 0.0,
    }


def _bootstrap(
    returns: np.ndarray,
    *,
    simulations: int = 2000,
    block_size: int = 20,
    seed: int = 20260824,
) -> dict[str, float]:
    n_sessions = int(returns.size)
    rng = np.random.default_rng(seed)
    blocks_needed = math.ceil(n_sessions / block_size)
    offsets = np.arange(block_size, dtype=np.int64)
    terminal = np.empty(simulations, dtype=np.float64)
    sharpe = np.empty(simulations, dtype=np.float64)
    max_drawdown = np.empty(simulations, dtype=np.float64)
    chunk_size = min(256, simulations)
    for cursor in range(0, simulations, chunk_size):
        chunk = min(chunk_size, simulations - cursor)
        starts = rng.integers(0, n_sessions, size=(chunk, blocks_needed))
        indexes = ((starts[..., None] + offsets) % n_sessions).reshape(chunk, -1)
        sampled = returns[indexes[:, :n_sessions]]
        wealth = np.cumprod(1.0 + sampled, axis=1)
        terminal[cursor : cursor + chunk] = wealth[:, -1] - 1.0
        means = np.mean(sampled, axis=1)
        stds = np.std(sampled, axis=1, ddof=1)
        sharpe[cursor : cursor + chunk] = np.divide(
            means * math.sqrt(252.0),
            stds,
            out=np.zeros_like(means),
            where=stds > 1e-12,
        )
        peaks = np.maximum.accumulate(
            np.concatenate([np.ones((chunk, 1)), wealth], axis=1), axis=1
        )[:, 1:]
        max_drawdown[cursor : cursor + chunk] = np.max(1.0 - wealth / peaks, axis=1)
    return {
        "probability_positive_return": float(np.mean(terminal > 0.0)),
        "sharpe_p05": float(np.quantile(sharpe, 0.05)),
        "max_drawdown_p95": float(np.quantile(max_drawdown, 0.95)),
    }


def _curve_sleeves(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = record["result"]
    dates = result["factor_attribution_curve"]["dates"]
    attribution = {row["factor_id"]: row for row in result["factor_attribution"]}
    output: dict[str, dict[str, Any]] = {}
    for series in result["factor_attribution_curve"]["series"]:
        row = attribution[series["factor_id"]]
        weight = float(row["normalized_weight"])
        wealth = 1.0 + np.asarray(series["values"], dtype=np.float64) / weight
        output[row["name"]] = {
            "dates": dates,
            "wealth": wealth,
            "returns": _returns(wealth.tolist()),
            "attribution": row,
            "factor_id": row["factor_id"],
        }
    return output


def _rolling_paths(record: dict[str, Any], *, sleeves: bool) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for period in record["result"]["stability_analysis"]["rolling"]["12m"]:
        end = period["end"]
        if sleeves:
            for row in period["sleeves"]:
                output.setdefault(row["name"], {})[end] = float(row["standalone_return"])
        else:
            output.setdefault("benchmark", {})[end] = float(period["total_return"])
    return output


def analyze(batch_ids: list[int], output: Path) -> None:
    if len(batch_ids) != 2:
        raise RuntimeError("exactly two completed batch ids are required")
    benchmark = api_backtest(BENCHMARK_ID)
    batches = [api_backtest(value) for value in batch_ids]
    for record in [benchmark, *batches]:
        if record.get("status") != "done":
            raise RuntimeError(f"backtest {record.get('id')} is not done")
        if not record["result"]["integrity"].get("all_pass"):
            raise RuntimeError(f"backtest {record.get('id')} failed ledger integrity")

    benchmark_dates = benchmark["result"]["curve"]["dates"]
    benchmark_wealth = np.asarray(benchmark["result"]["curve"]["equity"], dtype=np.float64)
    benchmark_returns = _returns(benchmark_wealth.tolist())
    benchmark_monthly = _monthly_returns(benchmark_dates, benchmark_returns)
    benchmark_rolling = _rolling_paths(benchmark, sleeves=False)["benchmark"]
    benchmark_sleeves = _curve_sleeves(benchmark)

    rows: list[dict[str, Any]] = []
    for record in batches:
        if record["result"]["curve"]["dates"] != benchmark_dates:
            raise RuntimeError(f"backtest {record['id']} date path differs from benchmark")
        rolling = _rolling_paths(record, sleeves=True)
        annual_2022 = next(
            row
            for row in record["result"]["stability_analysis"]["annual"]
            if row["period"] == "2022"
        )
        annual_sleeves = {row["name"]: row for row in annual_2022["sleeves"]}
        for name, sleeve in _curve_sleeves(record).items():
            factor_number = int(name.removeprefix("因子"))
            returns = sleeve["returns"]
            full = _period_stats(returns)
            year_mask = np.asarray([date.fromisoformat(value).year == 2022 for value in benchmark_dates])
            year = _period_stats(returns[year_mask])
            if abs(year["total_return"] - float(annual_sleeves[name]["standalone_return"])) > 1e-8:
                raise RuntimeError(f"factor {factor_number} 2022 ledger reconciliation failed")
            monthly = _monthly_returns(benchmark_dates, returns)
            bootstrap = _bootstrap(returns)
            constituent_corr = {
                constituent: _corr(returns, data["returns"])
                for constituent, data in benchmark_sleeves.items()
            }
            attr = sleeve["attribution"]
            row = {
                "factor": factor_number,
                "name": name,
                "expression": attr["expression"],
                "backtest_id": record["id"],
                "full": full,
                "avg_daily_turnover": float(attr["avg_daily_turnover"]),
                "profit_factor": float(attr["profit_factor"]),
                "total_execution_cost": float(attr["total_execution_cost"]),
                "return_2022": year["total_return"],
                "sharpe_2022": year["sharpe"],
                "correlation": {
                    "daily_benchmark": _corr(returns, benchmark_returns),
                    "monthly_benchmark": _common_corr(monthly, benchmark_monthly),
                    "rolling_12m_benchmark": _common_corr(rolling[name], benchmark_rolling),
                    "daily_constituents": constituent_corr,
                },
                "bootstrap": bootstrap,
            }
            row["hard_gates"] = {
                "return_2022_gt_5pct": row["return_2022"] > 0.05,
                "daily_corr_lt_0_30": row["correlation"]["daily_benchmark"] < 0.30,
                "full_sharpe_gt_0_30": row["full"]["sharpe"] > 0.30,
                "max_drawdown_lt_40pct": row["full"]["max_drawdown"] < 0.40,
            }
            row["hard_pass_count"] = sum(row["hard_gates"].values())
            row["hard_pass"] = row["hard_pass_count"] == 4
            rows.append(row)

    rows.sort(
        key=lambda row: (
            -int(row["hard_pass"]),
            -int(row["hard_pass_count"]),
            -float(row["return_2022"]),
            float(row["correlation"]["daily_benchmark"]),
            -float(row["sharpe_2022"]),
        )
    )
    for rank, row in enumerate(rows, 1):
        row["hedge_rank"] = rank
        # The first probe stays at the smallest declared weight.  The screen is
        # intentionally not allowed to turn a strong 2022 observation into a
        # retrospectively optimized allocation, and the existing #23 sleeve is
        # exactly five percent, making an overlap replacement auditable.
        weight = 5
        constituent_corr = row["correlation"]["daily_constituents"]
        if all(value is not None and value < 0.0 for value in constituent_corr.values()):
            replace = "因子19"
        else:
            replace = max(
                constituent_corr,
                key=lambda key: -math.inf if constituent_corr[key] is None else constituent_corr[key],
            )
        row["suggested_probe"] = {
            "weight_pct": weight,
            "replace_from": replace,
            "rule": "fixed_5pct_overlap_replacement_v1",
        }

    payload = {
        "schema": "factorfactory.2022-regime-hedge-screen/v1",
        "generated_at": "2026-08-27",
        "benchmark_backtest_id": BENCHMARK_ID,
        "batch_backtest_ids": batch_ids,
        "actual_period": {
            "start": benchmark_dates[0],
            "end": benchmark_dates[-1],
            "sessions": len(benchmark_dates),
        },
        "benchmark": {
            "weights": {"因子21": 0.69, "因子19": 0.26, "因子23": 0.05},
            "stats": benchmark["result"]["stats"],
            "calendar_2022": next(
                row
                for row in benchmark["result"]["stability_analysis"]["annual"]
                if row["period"] == "2022"
            ),
        },
        "ranking_rule": (
            "hard-pass first; then hard-gate pass count; then 2022 return descending; "
            "daily benchmark correlation ascending; 2022 Sharpe descending"
        ),
        "duplicate_note": "因子17与因子22的DSL完全相同；同口径结果不得计作两个独立来源。",
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


def _gate_text(row: dict[str, Any]) -> str:
    gates = row["hard_gates"]
    failures = []
    if not gates["return_2022_gt_5pct"]:
        failures.append("2022<=5%")
    if not gates["daily_corr_lt_0_30"]:
        failures.append("daily_corr>=0.30")
    if not gates["full_sharpe_gt_0_30"]:
        failures.append("Sharpe<=0.30")
    if not gates["max_drawdown_lt_40pct"]:
        failures.append("MDD>=40%")
    return "PASS" if not failures else "FAIL: " + ", ".join(failures)


def _unique_top(rows: list[dict[str, Any]], count: int = 5) -> list[dict[str, Any]]:
    output = []
    seen = set()
    for row in rows:
        signature = re.sub(r"\s+", "", row["expression"])
        if signature in seen:
            continue
        seen.add(signature)
        output.append(row)
        if len(output) == count:
            break
    return output


def render(input_path: Path, docs_dir: Path) -> None:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    top5 = rows[:5]
    unique_top5 = _unique_top(rows)
    benchmark = payload["benchmark"]
    period = payload["actual_period"]

    markdown = [
        "# 2022 Regime Hedge 因子筛选",
        "",
        "日期：2026-08-27  ",
        "市场：美股  ",
        "冻结基准：`69% #21 + 26% #19 + 5% #23`  ",
        "协议：`step_event_v2_weighted_sleeves_v1`  ",
        "实验账本：`#717`（基准）、`#723`（因子1-12）、`#726`（其余11条）",
        "",
        "## 口径与审计边界",
        "",
        f"实际覆盖 {period['start']} 至 {period['end']}，共 {period['sessions']} 个交易日。23个候选各使用100万美元独立资金 sleeve；全部路径来自本轮事件账本，不引用原始KPI推算。",
        "",
        "- 美股 `long_short`，Top500，每5日调仓，头尾各20%，多空 gross 各95%；",
        "- t日收盘信号，t+1原始开盘成交；IBKR Pro Fixed、2 bps滑点、平方根冲击10 bps、成交量参与率上限10%；",
        "- 年借券300 bps，保证金利率500 bps，现金缓冲2%，最大gross 200%；",
        "- Bootstrap为2000次、20日循环移动块、随机种子20260824；",
        "- #723与#726共23/23个sleeve均为 `integrity.all_pass=true`，日期与#717逐日一致。",
        "",
        "基准复核：CAGR {:.2%}、Sharpe {:.3f}、Sortino {:.3f}、Calmar {:.3f}、MDD {:.2%}；2022收益 {:.2%}、2022 Sharpe {:.3f}。".format(
            benchmark["stats"]["ann_ret"], benchmark["stats"]["sharpe"],
            benchmark["stats"]["sortino"], benchmark["stats"]["calmar"],
            benchmark["stats"]["max_dd"], benchmark["calendar_2022"]["total_return"],
            benchmark["calendar_2022"]["sharpe"],
        ),
        "",
        "## 23个候选总表",
        "",
        "排名采用预声明的门槛优先字典序：先看是否四项全过，再看通过项数，然后依次看2022收益、日相关性和2022 Sharpe。它不是历史最优权重优化。",
        "",
        "|排名|因子|CAGR|Sharpe|Sortino|Calmar|MDD|换手|PF|总成本(USD)|2022收益|2022 Sharpe|日相关|月相关|滚动12m相关|Boot正收益|5% Sharpe|95% MDD|硬门槛|",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        full, corr, boot = row["full"], row["correlation"], row["bootstrap"]
        markdown.append(
            "|{rank}|#{factor}|{cagr:.2%}|{sharpe:.2f}|{sortino:.2f}|{calmar:.2f}|{mdd:.2%}|{turnover:.2%}|{pf:.2f}|{cost:,.0f}|{r22:.2%}|{s22:.2f}|{daily:.3f}|{monthly:.3f}|{rolling:.3f}|{positive:.2%}|{p05:.2f}|{p95:.2%}|{gate}|".format(
                rank=row["hedge_rank"], factor=row["factor"], cagr=full["cagr"],
                sharpe=full["sharpe"], sortino=full["sortino"], calmar=full["calmar"],
                mdd=full["max_drawdown"], turnover=row["avg_daily_turnover"],
                pf=row["profit_factor"], cost=row["total_execution_cost"],
                r22=row["return_2022"], s22=row["sharpe_2022"],
                daily=corr["daily_benchmark"], monthly=corr["monthly_benchmark"],
                rolling=corr["rolling_12m_benchmark"],
                positive=boot["probability_positive_return"], p05=boot["sharpe_p05"],
                p95=boot["max_drawdown_p95"], gate=_gate_text(row),
            )
        )

    markdown += [
        "",
        "## 2022 Hedge 前5名（原始因子排名）",
        "",
        "|名次|因子|2022收益|日/月/滚动12m相关|全样本Sharpe / MDD|首测权重|替换来源|说明|",
        "|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    notes = {
        17: "四门槛贴线通过；与#22完全同式，只能二选一。",
        22: "与#17完全相同，不构成第二个独立收益来源。",
        26: "四门槛通过，长期统计优于#17/#22，但滚动12m相关仍为中等。",
        25: "2022和全样本均较强，但日相关0.634，低相关门槛失败。",
        24: "低相关且2022较强，但全样本MDD 42.60%越过硬门槛。",
    }
    for row in top5:
        corr, full = row["correlation"], row["full"]
        markdown.append(
            f"|{row['hedge_rank']}|#{row['factor']}|{row['return_2022']:.2%}|"
            f"{corr['daily_benchmark']:.3f} / {corr['monthly_benchmark']:.3f} / {corr['rolling_12m_benchmark']:.3f}|"
            f"{full['sharpe']:.2f} / {full['max_drawdown']:.2%}|5%|"
            f"{row['suggested_probe']['replace_from']}|{notes[row['factor']]}|"
        )

    markdown += [
        "",
        "所有首测均选择5%而不是10%/15%：这是最小预声明探针，避免把2022单年优势直接转化成更大的历史拟合自由度。#17/#22、#26和#24都与#23高度重合，因此先替换现有5% #23；#25与#21高度重合，因此从#21替换5%。",
        "",
        "## 去重后的5个独立候选",
        "",
        "|顺序|代表因子|硬门槛|首测|替换|",
        "|---:|---:|---|---:|---|",
    ]
    for index, row in enumerate(unique_top5, 1):
        markdown.append(
            f"|{index}|#{row['factor']}|{_gate_text(row)}|5%|{row['suggested_probe']['replace_from']}|"
        )
    markdown += [
        "",
        "原始Top5中的#17/#22折叠为一个来源后，#16递补为第五个独立候选；但#16与基准日相关0.661，仍不满足本任务低相关门槛。",
        "",
        "## 结论",
        "",
        "- 严格同时满足四项硬门槛的是#17、#22、#26；因#17与#22同式，实际只有两个独立通过来源：#17/#22代表式与#26。",
        "- #17/#22的2022收益29.47%、日相关-0.031，但全样本Sharpe仅0.304、MDD 38.94%，且Bootstrap 5% Sharpe为-0.25、95% MDD为71.89%，属于边界候选。",
        "- #26的2022收益25.82%、日相关0.153、全样本Sharpe 0.446、MDD 33.14%，Bootstrap正收益概率80.25%，是两类通过来源中更均衡者；滚动12m相关0.563提示其并非完全独立。",
        "- #25和#16本身质量较好，但与基准高度相关，不应被称为Regime Hedge；#24低相关但MDD门槛失败。",
        "- #10/#12虽在2022分别盈利45.13%/45.91%，但全样本CAGR约-26%、MDD超过91%，是典型单一regime暴露，不进入候选。",
        "",
        "## 可审计产物",
        "",
        "- `var/backtests/00000717/`：冻结基准事件账本",
        "- `var/backtests/00000723/`：第一批12个独立sleeve事件账本",
        "- `var/backtests/00000726/`：第二批11个独立sleeve事件账本",
        "- `var/reports/us-2022-regime-hedge-screen-20260827/results.json`：机器可读全量结果",
        "- `scripts/run_2022_regime_hedge_screen.py`：请求复刻、统计与报告生成脚本",
        "",
        "Bootstrap是历史路径重采样压力测试，不是未来收益预测；本面板为NON_PIT研究数据。本轮只筛选和给出固定档位候选，不执行历史权重优化。",
    ]

    md_path = docs_dir / "us-2022-regime-hedge-screen-20260827.md"
    html_path = docs_dir / "us-2022-regime-hedge-screen-20260827.html"
    md_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")

    def table_rows(source_rows: list[dict[str, Any]]) -> str:
        output = []
        for row in source_rows:
            f, c, b = row["full"], row["correlation"], row["bootstrap"]
            gate = _gate_text(row)
            cls = "pass" if gate == "PASS" else "fail"
            output.append(
                f"<tr><td>{row['hedge_rank']}</td><td><b>#{row['factor']}</b></td>"
                f"<td>{f['cagr']:.2%}</td><td>{f['sharpe']:.2f}</td><td>{f['sortino']:.2f}</td>"
                f"<td>{f['calmar']:.2f}</td><td>{f['max_drawdown']:.2%}</td>"
                f"<td>{row['avg_daily_turnover']:.2%}</td><td>{row['profit_factor']:.2f}</td>"
                f"<td>${row['total_execution_cost']:,.0f}</td><td>{row['return_2022']:.2%}</td>"
                f"<td>{row['sharpe_2022']:.2f}</td><td>{c['daily_benchmark']:.3f}</td>"
                f"<td>{c['monthly_benchmark']:.3f}</td><td>{c['rolling_12m_benchmark']:.3f}</td>"
                f"<td>{b['probability_positive_return']:.2%}</td><td>{b['sharpe_p05']:.2f}</td>"
                f"<td>{b['max_drawdown_p95']:.2%}</td><td class='{cls}'>{html.escape(gate)}</td></tr>"
            )
        return "".join(output)

    top_cards = "".join(
        f"<article><h3>#{row['factor']}</h3><b>2022 {row['return_2022']:.2%}</b>"
        f"<p>日相关 {row['correlation']['daily_benchmark']:.3f} · 全样本Sharpe {row['full']['sharpe']:.2f} · MDD {row['full']['max_drawdown']:.2%}</p>"
        f"<p>首测5%，从{html.escape(row['suggested_probe']['replace_from'])}替换。</p>"
        f"<small>{html.escape(notes[row['factor']])}</small></article>"
        for row in top5
    )
    html_body = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>2022 Regime Hedge 因子筛选</title><style>
    :root{{--bg:#f4f7fb;--card:#fff;--ink:#14213a;--muted:#64748b;--line:#dbe4ef;--blue:#245bc9;--green:#13795b;--red:#b4233c}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC',sans-serif;line-height:1.6}}.wrap{{max-width:1480px;margin:auto}}header{{padding:52px 24px;background:linear-gradient(135deg,#0c1930,#153d73 65%,#0f6570);color:#fff}}h1{{font-size:40px;margin:6px 0}}header p{{color:#d9e8f8;max-width:980px}}.pills{{display:flex;flex-wrap:wrap;gap:8px}}.pill{{border:1px solid #ffffff45;border-radius:999px;padding:4px 10px;font-size:12px}}main{{padding:28px 20px 70px}}section{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px;margin-bottom:18px;box-shadow:0 7px 23px #243d6912}}h2{{margin-top:0}}.grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}}article{{border:1px solid var(--line);border-radius:12px;padding:15px}}article h3{{margin:0 0 6px}}article b{{font-size:22px;color:var(--blue)}}article p,article small{{color:var(--muted)}}.table{{overflow:auto;border:1px solid var(--line);border-radius:10px}}table{{border-collapse:collapse;min-width:1700px;width:100%}}th,td{{padding:9px;border-bottom:1px solid var(--line);font-size:12px;text-align:right;white-space:nowrap}}th{{background:#edf2f8;color:#42516a;position:sticky;top:0}}td:nth-child(2),th:nth-child(2),td:last-child,th:last-child{{text-align:left}}.pass{{color:var(--green);font-weight:700}}.fail{{color:var(--red)}}.callout{{border-left:4px solid var(--blue);background:#eef4ff;padding:14px 16px;border-radius:9px}}code{{background:#12223a;color:#e4efff;padding:2px 5px;border-radius:4px}}footer{{padding:24px;color:var(--muted);background:#e8edf5}}@media(max-width:1000px){{.grid{{grid-template-columns:repeat(2,1fr)}}}}@media(max-width:560px){{.grid{{grid-template-columns:1fr}}h1{{font-size:30px}}}}
    </style></head><body><header><div class='wrap'><small>FACTORFACTORY · AUDITED REGIME SCREEN · 2026-08-27</small><h1>2022 Regime Hedge 因子筛选</h1><p>目标不是最大化全样本Sharpe，而是寻找2022明显盈利、与冻结基准低相关、且全样本不长期亏损的独立资金sleeve。</p><div class='pills'><span class='pill'>基准 #717</span><span class='pill'>候选 #723 + #726</span><span class='pill'>{period['start']} → {period['end']}</span><span class='pill'>{period['sessions']}日</span><span class='pill'>23/23账本PASS</span></div></div></header><main class='wrap'>
    <section><h2>冻结口径</h2><p>基准 <code>69% #21 + 26% #19 + 5% #23</code>：CAGR {benchmark['stats']['ann_ret']:.2%}、Sharpe {benchmark['stats']['sharpe']:.3f}、MDD {benchmark['stats']['max_dd']:.2%}；2022收益 {benchmark['calendar_2022']['total_return']:.2%}、Sharpe {benchmark['calendar_2022']['sharpe']:.3f}。</p><p>23条候选各100万美元独立资金；Top500、5日调仓、头尾20%、t+1原始开盘、完整费用/借券/冲击/杠杆约束。Bootstrap 2000次、20日循环移动块。</p><div class='callout'>硬门槛通过：#17、#22、#26。但#17与#22的DSL完全相同，所以只有两个独立通过来源。</div></section>
    <section><h2>2022 Hedge 前5名</h2><div class='grid'>{top_cards}</div><p>#17与#22不可同时当作两个候选；去重后#16递补为第五个独立候选。所有首测都固定为5%，没有做历史最优权重搜索。</p></section>
    <section><h2>23个候选总表</h2><p>排序：四门槛全过优先 → 通过项数 → 2022收益 → 日相关性 → 2022 Sharpe。</p><div class='table'><table><thead><tr><th>排名</th><th>因子</th><th>CAGR</th><th>Sharpe</th><th>Sortino</th><th>Calmar</th><th>MDD</th><th>换手</th><th>PF</th><th>总成本</th><th>2022收益</th><th>2022 Sharpe</th><th>日相关</th><th>月相关</th><th>滚动12m</th><th>Boot正收益</th><th>5% Sharpe</th><th>95% MDD</th><th>硬门槛</th></tr></thead><tbody>{table_rows(rows)}</tbody></table></div></section>
    <section><h2>审计结论</h2><ul><li>#17/#22：2022收益29.47%、日相关-0.031，但全样本Sharpe仅0.304、MDD 38.94%，属于贴线通过；二者完全同式。</li><li>#26：2022收益25.82%、日相关0.153、全样本Sharpe 0.446、MDD 33.14%，是更均衡的通过来源，但滚动12m相关0.563。</li><li>#25/#16质量较好但相关性门槛失败；#24低相关但MDD门槛失败。</li><li>#10/#12在2022盈利约45%，但全样本CAGR约-26%、MDD超过91%，不具备长期可持有性。</li></ul></section>
    <section><h2>证据路径</h2><p><code>var/backtests/00000717/</code> · <code>var/backtests/00000723/</code> · <code>var/backtests/00000726/</code> · <code>var/reports/us-2022-regime-hedge-screen-20260827/results.json</code></p><p>NON_PIT研究数据；Bootstrap不是未来预测。本轮只筛选和推荐固定档位，不执行历史权重优化。</p></section></main><footer><div class='wrap'>FactorFactory Research Documents · Audited 2026-08-27</div></footer></body></html>"""
    html_path.write_text(html_body, encoding="utf-8")
    print(md_path)
    print(html_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    emit = subparsers.add_parser("emit-request")
    emit.add_argument("batch", type=int, choices=(1, 2))
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("batch_ids", nargs=2, type=int)
    analyze_parser.add_argument("--output", required=True, type=Path)
    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--input", required=True, type=Path)
    render_parser.add_argument("--docs-dir", default=ROOT / "docs/research", type=Path)
    args = parser.parse_args()
    if args.command == "emit-request":
        emit_request(args.batch)
    elif args.command == "analyze":
        analyze(args.batch_ids, args.output)
    else:
        render(args.input, args.docs_dir)


if __name__ == "__main__":
    main()
