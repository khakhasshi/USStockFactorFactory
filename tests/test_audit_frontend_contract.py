from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[1] / "frontend" / "app.js").read_text()


def test_factor_research_names_rank_correlations_without_relabeling_pearson():
    research = SOURCE.split("const FactorLibrary =", 1)[1].split("const BacktestView =", 1)[0]
    assert "RankIC均值" in research
    assert "RankICIR" in research
    assert "m.rank_ic_mean ?? m.ic_mean" in research
    assert "m.rank_icir ?? m.icir" in research
    assert "<th>IC均值</th>" not in research
    assert "<th>ICIR</th>" not in research
    # Event diagnostic computes genuine Pearson and Spearman separately.
    backtest = SOURCE.split("const BacktestView =", 1)[1]
    assert "<th>IC</th><th>ICIR</th><th>RankIC</th><th>RankICIR</th>" in backtest
    assert "row.overall?.ic_mean" in backtest
    assert "row.overall?.rank_ic_mean" in backtest


def test_factor_detail_exposes_event_snapshot_and_overfit_blockers():
    for contract in ["validation.event_audit", "validation.audit_provenance",
                     "validation.overfit_governance", "immutable_inputs_available",
                     "missing_evidence_trials", "dsr_probability", "pbo?.pbo",
                     "actual_sessions", "code_sha256", "data_sha256",
                     "expression_sha256", "promotion_blockers"]:
        assert contract in SOURCE
    assert '"NOT_RUN"' in SOURCE
    assert "原始试验路径" in SOURCE


def test_trade_statistics_disclose_closed_lots_and_open_positions():
    assert "已平仓批次 / 胜率" in SOURCE
    assert "closed_lot_win_rate" in SOURCE
    assert "closed_lot_profit_factor" in SOURCE
    assert "open_unrealized_pnl_after_entry_fees" in SOURCE
    assert "unallocated_financing_cost" in SOURCE
    assert "不是完整持仓生命周期计数" in SOURCE
    assert "(result.stats.closed_lots ?? result.stats.closed_trades) === 0 ? null" in SOURCE
    assert "完整交易 / 胜率" not in SOURCE
