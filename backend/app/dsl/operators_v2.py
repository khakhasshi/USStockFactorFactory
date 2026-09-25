"""Explicit v2 operators. Legacy spellings retain their historical arithmetic."""
import numpy as np
import polars as pl

DSL_REVISION = "factorfactory.dsl/v2-20260925"
WINDOW_ARGUMENTS = {
    "ts_corr_v2": 2, "ts_beta": 2, "ts_residual": 2, "ts_mean_if": 2,
    **{n: 1 for n in ("ts_count", "ts_median", "ts_mad", "ts_robust_zscore",
        "ts_decay_linear", "ts_ewm", "ts_drawdown", "ts_bars_since", "ts_streak")},
}
OPERATORS_V2 = {
    "ts_corr_v2(x, y, w)": "标准Pearson相关；完整配对窗口；零方差返回null；替代新提案中的旧ts_corr",
    "where(cond, a, b)": "非零条件选a，否则b；条件缺失返回null；只是信号分支，不代表平仓指令",
    "ts_count(cond, w)": "完整w日窗口非零条件次数；缺失不是false",
    "ts_mean_if(x, cond, w)": "完整有效w日窗口内条件非零的x均值；无命中返回null",
    "ts_beta(y, x, w)": "滚动OLS y=a+beta*x的beta；完整配对窗口，x零方差返回null",
    "ts_residual(y, x, w)": "上述含截距OLS在当前日的残差；不是ts_resi的时间趋势残差",
    "ts_median(x, w)": "完整w日窗口中位数",
    "ts_mad(x, w)": "窗口内abs(x-同一窗口中位数)的中位数，未乘尺度常数",
    "ts_robust_zscore(x, w)": "(当前x-窗口中位数)/(1.4826*MAD)；MAD为0返回null",
    "ts_decay_linear(x, w)": "完整w日窗口按旧到新1..w线性加权均值",
    "ts_ewm(x, w)": "有限w日指数加权均值，alpha=2/(w+1)，权重归一；不是无限历史递归EMA",
    "ts_drawdown(x, w)": "当前正价格/过去w日最高正价格-1，非窗口最大回撤；非正输入返回null",
    "ts_bars_since(cond, w)": "完整w日窗口距最近非零条件的天数，今天为0，无命中返回null",
    "ts_streak(cond, w)": "完整w日窗口截至今日连续非零次数，最大w，今日false为0",
    "cs_residual(y, x)": "逐日截面含截距OLS残差；只用配对有效行，至少3行且x有方差",
    "group_rank(x, g)": "逐日同组平均并列百分位rank；g为已有字段/离散表达式，缺组或不足2行返回null",
    "group_zscore(x, g)": "逐日同组样本标准化；缺组、少于2行或零方差返回null",
}
NAMES = frozenset(k.split("(")[0] for k in OPERATORS_V2)


def upgrade_correlation(expression: str) -> str:
    """Only for NEW proposals: explicit textual migration, never stored replay."""
    import re
    return re.sub(r"\bts_corr\s*\(", "ts_corr_v2(", expression)


def build_v2(b, fn, args, win):
    if fn not in NAMES:
        return None
    arity = 3 if fn in {"where", "ts_corr_v2", "ts_beta", "ts_residual", "ts_mean_if"} else 2
    if len(args) != arity:
        raise ValueError(f"{fn} 需要 {arity} 个参数")
    def clean(node):
        x = b._mat(b.build(node).cast(pl.Float64))
        return b._mat(pl.when(x.is_finite()).then(x).otherwise(None))
    def roll(expr):
        return b._mat(expr.over(partition_by="ts_code", order_by="trade_date"))
    if fn == "where":
        cond = clean(args[0])
        return b._mat(pl.when(cond.is_null()).then(None).when(cond != 0)
                      .then(b.build(args[1])).otherwise(b.build(args[2])))
    if fn in {"group_rank", "group_zscore"}:
        x, group = clean(args[0]), clean(args[1])
        x = b._mat(pl.when(group.is_not_null()).then(x).otherwise(None))
        keys = ["trade_date", group.meta.output_name()]
        n = b._mat(x.count().over(keys))
        if fn == "group_rank":
            result = b._mat(x.rank(method="average").over(keys)) / n
        else:
            mean, std = b._mat(x.mean().over(keys)), b._mat(x.std(ddof=1).over(keys))
            result = pl.when(std > 0).then((x - mean) / std).otherwise(None)
        return b._mat(pl.when(n >= 2).then(result).otherwise(None))
    if fn == "cs_residual":
        y, x = clean(args[0]), clean(args[1])
        valid = x.is_not_null() & y.is_not_null()
        x, y = b._mat(pl.when(valid).then(x)), b._mat(pl.when(valid).then(y))
        mx, my = b._mat(x.mean().over("trade_date")), b._mat(y.mean().over("trade_date"))
        dx, dy = b._mat(x - mx), b._mat(y - my)
        variance = b._mat((dx * dx).mean().over("trade_date"))
        cov = b._mat((dx * dy).mean().over("trade_date"))
        count = b._mat(x.count().over("trade_date"))
        return b._mat(pl.when((count >= 3) & (variance > 0)).then(dy - cov / variance * dx).otherwise(None))
    w = win(args[WINDOW_ARGUMENTS[fn]])
    if fn in {"ts_corr_v2", "ts_beta", "ts_residual"}:
        if w < 2:
            raise ValueError(f"{fn} 窗口必须至少为2")
        y, x = clean(args[0]), clean(args[1])
        valid = x.is_not_null() & y.is_not_null()
        x, y = b._mat(pl.when(valid).then(x)), b._mat(pl.when(valid).then(y))
        # Center each window BEFORE multiplication. E[xy]-E[x]E[y] (including
        # native rolling covariance) loses the signal at large price offsets.
        def regression(series):
            xv = np.asarray(series.struct.field("x").to_numpy(), dtype=float)
            yv = np.asarray(series.struct.field("y").to_numpy(), dtype=float)
            out = np.full(len(xv), np.nan)
            if len(xv) < w: return pl.Series(out).fill_nan(None)
            xs = np.lib.stride_tricks.sliding_window_view(xv, w)
            ys = np.lib.stride_tricks.sliding_window_view(yv, w)
            for start in range(0, len(xs), 4096):
                a, c = xs[start:start+4096], ys[start:start+4096]
                valid = np.isfinite(a).all(axis=1) & np.isfinite(c).all(axis=1)
                a, c = a-a[:, :1], c-c[:, :1]
                a, c = a-a.mean(axis=1)[:, None], c-c.mean(axis=1)[:, None]
                xx, yy, xy = (a*a).sum(axis=1), (c*c).sum(axis=1), (a*c).sum(axis=1)
                denom = np.sqrt(xx)*np.sqrt(yy) if fn == "ts_corr_v2" else xx
                result = np.divide(xy, denom, out=np.full(len(a),np.nan), where=valid & (denom>0))
                if fn == "ts_corr_v2": result = np.clip(result,-1.,1.)
                if fn == "ts_residual":
                    result = c[:, -1] - result*a[:, -1]
                    tolerance = 32*np.finfo(float).eps*np.maximum(1.,np.max(abs(c),axis=1))
                    result = np.where(abs(result)<=tolerance,0.,result)
                out[w-1+start:w-1+start+len(a)] = np.where(valid,result,np.nan)
            return pl.Series(out).fill_nan(None)
        return roll(pl.struct(x.alias("x"),y.alias("y")).map_batches(regression,return_dtype=pl.Float64))
    x = clean(args[0])
    if fn == "ts_mean_if":
        cond = clean(args[1])
        valid = x.is_not_null() & cond.is_not_null()
        hits = b._mat(pl.when(valid).then((cond != 0).cast(pl.Float64)).otherwise(None))
        selected = b._mat(pl.when(valid).then(pl.when(cond != 0).then(x).otherwise(0.)).otherwise(None))
        n, total = roll(hits.rolling_sum(w, min_samples=w)), roll(selected.rolling_sum(w, min_samples=w))
        return b._mat(pl.when(n > 0).then(total / n).otherwise(None))
    if fn == "ts_count":
        return roll((x != 0).cast(pl.Float64).rolling_sum(w, min_samples=w))
    if fn == "ts_median": return roll(x.rolling_median(w, min_samples=w))
    if fn in {"ts_decay_linear", "ts_ewm"}:
        weights = np.arange(1, w + 1, dtype=float) if fn == "ts_decay_linear" else (1 - 2/(w+1)) ** np.arange(w-1, -1, -1, dtype=float)
        weights = (weights / weights.sum()).tolist()
        # Polars weighted rolling panics on arrays with nulls. Zero-fill only
        # inside the numerical kernel, then restore strict full-window validity.
        count = roll(x.is_not_null().cast(pl.Int64).rolling_sum(w, min_samples=w))
        filled = b._mat(x.fill_null(0.))
        mean = roll(filled.rolling_mean(w, weights=weights, min_samples=w))
        return b._mat(pl.when(count == w).then(mean).otherwise(None))
    if fn == "ts_drawdown":
        positive = b._mat(pl.when(x > 0).then(x).otherwise(None))
        return b._mat(positive / roll(positive.rolling_max(w, min_samples=w)) - 1.)
    # Nonlinear bounded-window statistics. Chunk medians to avoid N*w copies.
    def windows(series):
        values = np.asarray(series.to_numpy(), dtype=float)
        out = np.full(len(values), np.nan)
        if len(values) < w: return pl.Series(out).fill_nan(None)
        view = np.lib.stride_tricks.sliding_window_view(values, w)
        for start in range(0, len(view), 4096):
            chunk = view[start:start+4096]
            valid = np.isfinite(chunk).all(axis=1)
            if fn in {"ts_mad", "ts_robust_zscore"}:
                med = np.median(chunk, axis=1)
                mad = np.median(np.abs(chunk - med[:, None]), axis=1)
                result = mad
                if fn == "ts_robust_zscore":
                    result = np.divide(chunk[:, -1] - med, 1.4826 * mad,
                        out=np.full(len(chunk), np.nan), where=mad > 0)
            else:
                truth = chunk != 0
                if fn == "ts_bars_since":
                    result = np.where(truth.any(axis=1), np.argmax(truth[:, ::-1], axis=1), np.nan)
                else:
                    result = np.where(truth.all(axis=1), w, np.argmax(~truth[:, ::-1], axis=1)).astype(float)
            out[w-1+start:w-1+start+len(chunk)] = np.where(valid, result, np.nan)
        return pl.Series(out).fill_nan(None)
    return roll(x.map_batches(windows, return_dtype=pl.Float64))
