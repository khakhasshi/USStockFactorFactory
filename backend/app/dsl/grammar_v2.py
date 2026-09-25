"""Field-aware v2 search shapes shared by structured search and ML pools."""
def candidates(family, fields, w=20):
    f = set(fields)
    if not {"close", "open", "high", "low", "vol", "amount"}.issubset(f): return []
    r = "returns(close,1)"
    cond = f"gt({r},0)"
    trend = f"returns(close,{w})"
    volume = f"vol/(ts_mean(vol,{w})+1e-9)"
    shapes = {
        "momentum": [f"rank(ts_decay_linear({r},{w}))", f"rank(ts_ewm({r},{w}))",
            f"rank(where(gt({trend},0),{trend},ts_mean_if({r},{cond},{w})))",
            f"rank(ts_streak({cond},{w})*{trend})"],
        "reversal": [f"-rank(ts_robust_zscore({trend},{w}))",
            f"-rank(ts_mean_if({r},lt({r},0),{w}))",
            f"rank(ts_drawdown(close,{w})*ts_bars_since({cond},{w}))"],
        "volatility": [f"-rank(ts_mad({r},{w}))", f"-rank(ts_median(abs({r}),{w}))",
            f"-rank(ts_count(lt({r},0),{w})*ts_std({r},{w}))"],
        "liquidity": [f"rank(ts_robust_zscore(amount,{w}))", f"rank(ts_mean_if({volume},lt({r},0),{w}))"],
        "volume_price_interaction": [f"rank(ts_corr_v2({r},{volume},{w}))",
            f"rank(ts_beta({r},{volume},{w}))", f"rank(ts_residual({r},{volume},{w}))",
            f"cs_residual(rank({trend}),rank(amount))",
            f"group_rank({trend},gt({volume},1))", f"group_zscore({trend},gt({volume},1))"],
        "gap_intraday": [f"rank(ts_mean_if((open-delay(close,1))/delay(close,1),gt(close,open),{w}))"],
        "price_relationship": [f"rank(ts_corr_v2(high/close,low/close,{w}))", f"rank(ts_residual(high/close,low/close,{w}))"],
    }
    if "pb" in f: shapes["valuation"] = [f"-rank(ts_robust_zscore(pb,{w}))", "group_rank(-pb,gt(returns(close,20),0))"]
    if "total_mv" in f: shapes["size"] = [f"-rank(ts_median(total_mv,{w}))"]
    if "net_mf_amount" in f: shapes["capital_flow"] = [f"rank(ts_mean_if(net_mf_amount/amount,lt({r},0),{w}))"]
    return shapes.get(family, [])
