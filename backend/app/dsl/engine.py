"""因子表达式 DSL: Python 语法子集 -> polars 多阶段流水线.

安全性: 仅白名单 AST 节点/算子/字段; 所有时序算子仅向后看, 结构上无前视.
实现要点: polars 不支持嵌套窗口表达式 (over 内含 over 会静默产生 null),
因此每个窗口算子的结果都物化为临时列, 保证任何阶段内 over 均为单层。
"""

import ast
import hashlib
import math
import re

import numpy as np
import polars as pl

from ..config import DSL_FIELDS
from .operators_v2 import DSL_REVISION, OPERATORS_V2, WINDOW_ARGUMENTS, build_v2, upgrade_correlation

_BY_CODE = {"partition_by": "ts_code", "order_by": "trade_date"}
MAX_EXPRESSION_LENGTH = 4000
MAX_EXPRESSION_AST_NODES = 1000


def _ts(expr: pl.Expr) -> pl.Expr:
    return expr.over(**_BY_CODE)


def _rolling_corr(x: pl.Expr, y: pl.Expr, w: int) -> pl.Expr:
    # corr = (E[xy]-E[x]E[y]) / (std_x*std_y), 全部用向后滚动统计实现
    mxy = (x * y).rolling_mean(w, min_samples=w)
    mx = x.rolling_mean(w, min_samples=w)
    my = y.rolling_mean(w, min_samples=w)
    sx = x.rolling_std(w, min_samples=w)
    sy = y.rolling_std(w, min_samples=w)
    return (mxy - mx * my) / (sx * sy + 1e-12)


def _win(node_w: ast.expr) -> int:
    # The native research grammar advertises 1..250 to the search agents.
    # Historical AutoAlpha libraries also contain conventional 251/252-session
    # one-year windows.  Accept those two compatibility values without
    # broadening the generated-search prompt or permitting arbitrary history.
    if not (
        isinstance(node_w, ast.Constant)
        and isinstance(node_w.value, int)
        and 1 <= node_w.value <= 252
    ):
        raise ValueError("窗口参数必须是 1..252 的整数字面量")
    return node_w.value


def _number(node: ast.expr, *, name: str) -> float:
    if not (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        raise ValueError(f"{name}必须是数值字面量")
    value = float(node.value)
    if not math.isfinite(value):
        raise ValueError(f"{name}必须是有限数值")
    return value


OPERATORS_DOC = {
    "ts_mean(x, w)": "w 日滚动均值",
    "ts_std(x, w)": "w 日滚动标准差",
    "ts_sum(x, w)": "w 日滚动求和",
    "ts_min(x, w)": "w 日滚动最小",
    "ts_max(x, w)": "w 日滚动最大",
    "ts_rank(x, w)": "当前值在过去 w 日中的分位",
    "ts_delta(x, w)": "x - delay(x, w)",
    "returns(x, w)": "x / delay(x, w) - 1",
    "ts_corr(x, y, w)": "LEGACY v1相关：分母样本标准差导致约(w-1)/w缩放；只供历史复现，新研究用ts_corr_v2",
    "ts_quantile(x, w, q)": "w 日滚动 q 分位数",
    "ts_slope(x, w)": "w 日线性回归斜率（Qlib 兼容）",
    "ts_rsquare(x, w)": "w 日线性回归 R²（Qlib 兼容）",
    "ts_resi(x, w)": "w 日线性回归当前残差（Qlib 兼容）",
    "ts_argmax(x, w)": "w 日窗口最大值首次出现位置，1..w（Qlib 兼容）",
    "ts_argmin(x, w)": "w 日窗口最小值首次出现位置，1..w（Qlib 兼容）",
    "delay(x, d)": "滞后 d 日",
    "maximum(x, y)": "逐元素较大值",
    "minimum(x, y)": "逐元素较小值",
    "gt(x, y)": "逐元素 x>y，输出 0/1",
    "lt(x, y)": "逐元素 x<y，输出 0/1",
    "rank(x)": "当日截面百分位排名",
    "zscore(x)": "当日截面 zscore",
    "winsor(x)": "当日截面 2.5 倍标准差截尾",
    "winsor_mad(x, threshold)": "当日截面 threshold 倍 MAD 截尾",
    "log(x)": "log(|x|+1e-9)",
    "abs(x)": "绝对值",
    "sign(x)": "符号",
}
OPERATORS_DOC.update(OPERATORS_V2)
PROPOSAL_OPERATORS_DOC = {k: v for k, v in OPERATORS_DOC.items() if not k.startswith("ts_corr(")}


class FactorPipeline:
    """有序临时列阶段 + 最终表达式; apply 后临时列被丢弃."""

    def __init__(self, stages: list[tuple[str, pl.Expr]], final: pl.Expr,
                 node_outputs: dict[str, pl.Expr] | None = None) -> None:
        self.stages = stages
        self.final = final
        self.node_outputs = node_outputs or {}

    def apply(self, lf: pl.LazyFrame, alias: str = "factor") -> pl.LazyFrame:
        for name, expr in self.stages:
            lf = lf.with_columns(expr.alias(name))
        lf = lf.with_columns(self.final.alias(alias))
        tmp = [n for n, _ in self.stages]
        return lf.drop(tmp) if tmp else lf


class _Builder:
    def __init__(self, fields: list[str] | None = None,
                 overrides: dict[str, str] | None = None) -> None:
        self.stages: list[tuple[str, pl.Expr]] = []
        self.fields = fields or DSL_FIELDS
        self.overrides = overrides or {}
        self.node_outputs: dict[str, pl.Expr] = {}

    def _mat(self, expr: pl.Expr) -> pl.Expr:
        name = f"__t{len(self.stages)}"
        self.stages.append((name, expr))
        return pl.col(name)

    def build(self, node: ast.expr) -> pl.Expr:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return pl.lit(float(node.value))
            raise ValueError(f"非法常量: {node.value!r}")
        if isinstance(node, ast.Name):
            if node.id not in self.fields:
                raise ValueError(f"未知字段: {node.id} (白名单: {self.fields})")
            return pl.col(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self.build(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = self.build(node.left), self.build(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / (right + 1e-12)
            raise ValueError(f"非法运算符: {type(node.op).__name__}")
        if isinstance(node, ast.Call):
            key = ast.dump(node, include_attributes=False)
            if key in self.overrides:
                return pl.col(self.overrides[key])
            result = self._call(node)
            self.node_outputs[key] = result
            return result
        raise ValueError(f"非法语法节点: {type(node).__name__}")

    def _call(self, node: ast.Call) -> pl.Expr:
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise ValueError("非法调用形式")
        fn = node.func.id
        a = node.args
        new_operator = build_v2(self, fn, a, _win)
        if new_operator is not None:
            return new_operator

        def arity(n: int) -> None:
            if len(a) != n:
                raise ValueError(f"{fn} 需要 {n} 个参数")

        # 无窗口的逐元素算子: 不需要物化
        if fn == "log":
            arity(1)
            return (self.build(a[0]).abs() + 1e-9).log()
        if fn == "abs":
            arity(1)
            return self.build(a[0]).abs()
        if fn == "sign":
            arity(1)
            return self.build(a[0]).sign()
        if fn in {"maximum", "minimum", "gt", "lt"}:
            arity(2)
            left, right = self.build(a[0]), self.build(a[1])
            if fn == "maximum":
                return pl.max_horizontal(left, right)
            if fn == "minimum":
                return pl.min_horizontal(left, right)
            comparison = left > right if fn == "gt" else left < right
            return comparison.cast(pl.Float64)

        # 窗口算子: 操作数先构建 (其内部窗口已物化为临时列), 结果整体再物化
        if fn == "delay":
            arity(2)
            return self._mat(_ts(self.build(a[0]).shift(_win(a[1]))))
        if fn == "returns":
            arity(2)
            periods = _win(a[1])
            x = self.build(a[0])
            lagged = self._mat(_ts(x.shift(periods)))
            return self._mat(x / (lagged + 1e-12) - 1.0)
        if fn in ("ts_mean", "ts_std", "ts_sum", "ts_min", "ts_max"):
            arity(2)
            w = _win(a[1])
            x = self.build(a[0])
            roll = {
                "ts_mean": x.rolling_mean, "ts_std": x.rolling_std, "ts_sum": x.rolling_sum,
                "ts_min": x.rolling_min, "ts_max": x.rolling_max,
            }[fn]
            return self._mat(_ts(roll(w, min_samples=w)))
        if fn == "ts_delta":
            arity(2)
            x = self.build(a[0])
            return self._mat(_ts(x - x.shift(_win(a[1]))))
        if fn == "ts_rank":
            arity(2)
            w = _win(a[1])
            x = self.build(a[0])
            # The former rolling_map callback calculated
            # count(window <= current) / window_size.  Native rolling_rank
            # with max-tie ranking is exactly that statistic, while remaining
            # inside Polars' vectorised engine instead of invoking Python once
            # for every security-date window.
            return self._mat(
                _ts(
                    x.rolling_rank(
                        w,
                        method="max",
                        min_samples=w,
                    )
                    / float(w)
                )
            )
        if fn == "ts_corr":
            arity(3)
            w = _win(a[2])
            return self._mat(_ts(_rolling_corr(self.build(a[0]), self.build(a[1]), w)))
        if fn == "ts_quantile":
            arity(3)
            w = _win(a[1])
            quantile = _number(a[2], name="ts_quantile q")
            if not 0.0 <= quantile <= 1.0:
                raise ValueError("ts_quantile q 必须在 [0, 1] 内")
            x = self.build(a[0])
            return self._mat(
                _ts(
                    x.rolling_quantile(
                        quantile,
                        interpolation="linear",
                        window_size=w,
                        min_samples=w,
                    )
                )
            )
        if fn in {"ts_slope", "ts_rsquare", "ts_resi"}:
            arity(2)
            w = _win(a[1])
            if w < 2:
                raise ValueError(f"{fn} 窗口必须至少为 2")
            x = self.build(a[0])
            sum_y = self._mat(_ts(x.rolling_sum(w, min_samples=w)))
            sum_y2 = self._mat(_ts((x * x).rolling_sum(w, min_samples=w)))
            sum_xy = self._mat(
                _ts(
                    x.fill_null(0.).rolling_sum(
                        w,
                        weights=[float(i) for i in range(1, w + 1)],
                        min_samples=w,
                    )
                )
            )
            n = float(w)
            sum_x = n * (n + 1.0) / 2.0
            sum_x2 = n * (n + 1.0) * (2.0 * n + 1.0) / 6.0
            numerator = n * sum_xy - sum_x * sum_y
            x_denominator = n * sum_x2 - sum_x * sum_x
            slope = self._mat(numerator / (x_denominator + 1e-12))
            if fn == "ts_slope":
                return slope
            y_denominator = n * sum_y2 - sum_y * sum_y
            if fn == "ts_rsquare":
                return self._mat(
                    (numerator * numerator)
                    / (x_denominator * y_denominator + 1e-12)
                )
            intercept = self._mat(sum_y / n - slope * (sum_x / n))
            return self._mat(x - (slope * n + intercept))
        if fn in {"ts_argmax", "ts_argmin"}:
            arity(2)
            w = _win(a[1])
            x = self.build(a[0])

            def rolling_positions(series: pl.Series) -> pl.Series:
                values = np.asarray(series.to_numpy(), dtype=float)
                output = np.full(values.shape[0], np.nan, dtype=float)
                if values.shape[0] < w:
                    return pl.Series(output)
                windows = np.lib.stride_tricks.sliding_window_view(values, w)
                valid = np.isfinite(windows).all(axis=1)
                if fn == "ts_argmax":
                    positions = np.argmax(windows, axis=1) + 1
                else:
                    positions = np.argmin(windows, axis=1) + 1
                output[w - 1 :] = np.where(valid, positions, np.nan)
                return pl.Series(output)

            return self._mat(
                _ts(
                    x.map_batches(
                        rolling_positions,
                        return_dtype=pl.Float64,
                    )
                )
            )
        if fn == "rank":
            arity(1)
            x = self.build(a[0])
            return self._mat(
                x.rank(method="average").over("trade_date") / (x.count().over("trade_date") + 1e-12)
            )
        if fn == "zscore":
            arity(1)
            x = self.build(a[0])
            return self._mat((x - x.mean().over("trade_date")) / (x.std().over("trade_date") + 1e-12))
        if fn == "winsor":
            arity(1)
            x = self.build(a[0])
            m, s = x.mean().over("trade_date"), x.std().over("trade_date")
            return self._mat(x.clip(m - 2.5 * s, m + 2.5 * s))
        if fn == "winsor_mad":
            arity(2)
            threshold = _number(a[1], name="winsor_mad threshold")
            if not 0 < threshold <= 20:
                raise ValueError("winsor_mad threshold 必须在 (0, 20] 内")
            x = self.build(a[0])
            median = self._mat(x.median().over("trade_date"))
            mad = self._mat(
                (x - median).abs().median().over("trade_date") * 1.4826
            )
            return self._mat(
                x.clip(
                    median - threshold * mad,
                    median + threshold * mad,
                )
            )
        raise ValueError(f"未知算子: {fn}")


def _degenerate_check(root: ast.expr) -> None:
    """拒绝数学退化子式 (恒常数经 rank 放大浮点噪声可刷分, 实验1已被随机搜索利用)."""
    for n in ast.walk(root):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in {"ts_beta", "ts_residual", "cs_residual"}:
            if len(n.args) >= 2 and ast.dump(n.args[0]) == ast.dump(n.args[1]):
                raise ValueError("退化表达式: 同一变量的回归恒为常数")
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Sub, ast.Div)):
            if ast.dump(n.left) == ast.dump(n.right):
                raise ValueError("退化表达式: x-x / x/x 恒为常数")
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in {"ts_corr", "ts_corr_v2"}:
            if len(n.args) >= 2 and ast.dump(n.args[0]) == ast.dump(n.args[1]):
                raise ValueError("退化表达式: ts_corr(x, x, w) 恒为 1")


def parse(expression: str, fields: list[str] | None = None, *,
          _overrides: dict[str, str] | None = None) -> FactorPipeline:
    """解析 DSL 表达式为 FactorPipeline; 非法即抛 ValueError."""
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError(f"表达式过长（最大 {MAX_EXPRESSION_LENGTH} 字符）")
    tree = ast.parse(expression, mode="eval")
    node_count = sum(1 for _ in ast.walk(tree))
    if node_count > MAX_EXPRESSION_AST_NODES:
        raise ValueError(
            f"表达式结构过于复杂（最大 {MAX_EXPRESSION_AST_NODES} 个语法节点）"
        )
    _degenerate_check(tree.body)
    # Overrides are internal materialization columns, never a validation bypass.
    if _overrides:
        _Builder(fields).build(tree.body)
    b = _Builder(fields, _overrides)
    final = b.build(tree.body)
    return FactorPipeline(b.stages, final, b.node_outputs)


def _flatten_root_multiplication(node: ast.expr) -> list[ast.expr]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return [
            *_flatten_root_multiplication(node.left),
            *_flatten_root_multiplication(node.right),
        ]
    return [node]


def _direction_invariant_root_key(node: ast.expr) -> str:
    """Canonicalise only a global sign; nonlinear nested signs stay distinct."""
    while isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        node = node.operand
    factors = _flatten_root_multiplication(node)
    if len(factors) == 1:
        return ast.dump(node, include_attributes=False)
    keys = []
    for factor in factors:
        while isinstance(factor, ast.UnaryOp) and isinstance(
            factor.op, ast.USub
        ):
            factor = factor.operand
        if (
            isinstance(factor, ast.Constant)
            and isinstance(factor.value, (int, float))
            and not isinstance(factor.value, bool)
            and abs(float(factor.value)) == 1.0
        ):
            continue
        keys.append(ast.dump(factor, include_attributes=False))
    if not keys:
        return "GLOBAL_SCALAR"
    if len(keys) == 1:
        return keys[0]
    return "ROOT_MUL(" + "|".join(sorted(keys)) + ")"


def normalize_hash(
    expression: str,
    *,
    direction_invariant: bool = False,
) -> str:
    tree = ast.parse(expression, mode="eval")
    key = (
        _direction_invariant_root_key(tree.body)
        if direction_invariant
        else ast.dump(tree, include_attributes=False)
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def validate(expression: str, fields: list[str] | None = None) -> str | None:
    try:
        parse(expression, fields)
        return None
    except (ValueError, SyntaxError) as e:
        return str(e)


def _history_for_node(node: ast.expr) -> int:
    """Conservative trading-session history required to evaluate one DSL node."""
    if isinstance(node, (ast.Constant, ast.Name)):
        return 1
    if isinstance(node, ast.UnaryOp):
        return _history_for_node(node.operand)
    if isinstance(node, ast.BinOp):
        return max(_history_for_node(node.left), _history_for_node(node.right))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = node.func.id
        child = max((_history_for_node(arg) for arg in node.args if not isinstance(arg, ast.Constant)), default=1)
        if fn in WINDOW_ARGUMENTS:
            return child + _win(node.args[WINDOW_ARGUMENTS[fn]])
        if fn in {"ts_quantile", "ts_slope", "ts_rsquare", "ts_resi", "ts_argmax", "ts_argmin"}:
            return child + _win(node.args[1])
        if fn in {"delay", "returns"} and len(node.args) == 2:
            return child + _win(node.args[1])
        if fn in {"ts_mean", "ts_std", "ts_sum", "ts_min", "ts_max", "ts_rank", "ts_delta"} \
                and len(node.args) == 2:
            return child + _win(node.args[1])
        if fn == "ts_corr" and len(node.args) == 3:
            return child + _win(node.args[2])
        return child
    return 1


def required_history(expression: str) -> int:
    """Return a safe lookback count used by fast point-in-time screening."""
    tree = ast.parse(expression, mode="eval")
    _degenerate_check(tree.body)
    return max(1, _history_for_node(tree.body))


def _latex_name(name: str) -> str:
    escaped = name.replace("_", r"\_")
    return rf"\mathrm{{{escaped}}}"


def _latex_number(value: int | float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _latex_for_node(node: ast.expr, parent_precedence: int = 0) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return _latex_number(node.value)
    if isinstance(node, ast.Name):
        return _latex_name(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _latex_for_node(node.operand, 30)
        return rf"-{inner}"
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Div):
            return rf"\frac{{{_latex_for_node(node.left)}}}{{{_latex_for_node(node.right)}}}"
        if isinstance(node.op, ast.Mult):
            value = rf"{_latex_for_node(node.left, 20)} \cdot {_latex_for_node(node.right, 20)}"
            precedence = 20
        elif isinstance(node.op, ast.Add):
            value = rf"{_latex_for_node(node.left, 10)} + {_latex_for_node(node.right, 10)}"
            precedence = 10
        elif isinstance(node.op, ast.Sub):
            value = rf"{_latex_for_node(node.left, 10)} - {_latex_for_node(node.right, 11)}"
            precedence = 10
        else:
            raise ValueError(f"无法转换的运算符: {type(node.op).__name__}")
        return rf"\left({value}\right)" if precedence < parent_precedence else value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = node.func.id
        args = node.args
        rendered = [_latex_for_node(arg) for arg in args]
        if fn == "rank":
            return rf"\operatorname{{Rank}}_{{cs}}\left({rendered[0]}\right)"
        if fn == "zscore":
            return rf"\operatorname{{ZScore}}_{{cs}}\left({rendered[0]}\right)"
        if fn == "winsor":
            return rf"\operatorname{{Winsor}}_{{2.5\sigma}}\left({rendered[0]}\right)"
        if fn == "winsor_mad":
            return (
                rf"\operatorname{{WinsorMAD}}_{{{rendered[1]}}}"
                rf"\left({rendered[0]}\right)"
            )
        if fn == "log":
            return rf"\log\left(\left|{rendered[0]}\right|+\epsilon\right)"
        if fn == "abs":
            return rf"\left|{rendered[0]}\right|"
        if fn == "sign":
            return rf"\operatorname{{sgn}}\left({rendered[0]}\right)"
        if fn == "delay":
            return rf"\operatorname{{Delay}}_{{{rendered[1]}}}\left({rendered[0]}\right)"
        if fn == "returns":
            return rf"\operatorname{{Return}}_{{{rendered[1]}}}\left({rendered[0]}\right)"
        if fn == "ts_delta":
            return rf"\Delta_{{{rendered[1]}}}\left({rendered[0]}\right)"
        if fn == "ts_corr":
            return (
                rf"\operatorname{{Corr}}_{{{rendered[2]}}}"
                rf"\left({rendered[0]},\,{rendered[1]}\right)"
            )
        rolling = {
            "ts_mean": "Mean",
            "ts_std": "Std",
            "ts_sum": "Sum",
            "ts_min": "Min",
            "ts_max": "Max",
            "ts_rank": "Rank",
        }
        if fn in rolling:
            return rf"\operatorname{{{rolling[fn]}}}_{{{rendered[1]}}}\left({rendered[0]}\right)"
        if any(k.startswith(fn + "(") for k in OPERATORS_DOC):
            return _latex_name(fn) + r"\left(" + ",\\,".join(rendered) + r"\right)"
        raise ValueError(f"无法转换的 DSL 算子: {fn}")
    raise ValueError(f"无法转换的语法节点: {type(node).__name__}")


def expression_to_latex(expression: str) -> str:
    """Translate a validated DSL expression to safe KaTeX-compatible LaTeX."""
    tree = ast.parse(expression, mode="eval")
    _degenerate_check(tree.body)
    return _latex_for_node(tree.body)


def expression_profile(expression: str) -> dict:
    """Return cheap structural metadata used by the factor library and UI."""
    tree = ast.parse(expression, mode="eval")
    _degenerate_check(tree.body)
    operators: list[str] = []
    fields: list[str] = []
    windows: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            operators.append(node.func.id)
            if node.func.id in WINDOW_ARGUMENTS:
                windows.append(_win(node.args[WINDOW_ARGUMENTS[node.func.id]]))
            if node.func.id in {
                "delay",
                "returns",
                "ts_mean",
                "ts_std",
                "ts_sum",
                "ts_min",
                "ts_max",
                "ts_rank",
                "ts_delta",
                "ts_corr",
            }:
                for arg in node.args[1:]:
                    if (
                        isinstance(arg, ast.Constant)
                        and isinstance(arg.value, int)
                    ):
                        windows.append(arg.value)
        elif isinstance(node, ast.Name) and not isinstance(getattr(node, "ctx", None), ast.Load):
            fields.append(node.id)
        elif isinstance(node, ast.Name):
            # Function names also appear as ast.Name; remove them below.
            fields.append(node.id)
    fields = [name for name in fields if name not in set(operators)]
    canonical = re.sub(r"\s+", "", ast.unparse(tree.body))
    return {
        "dsl_revision": DSL_REVISION,
        "correlation_semantics": "mixed_explicit_v1_v2" if {"ts_corr", "ts_corr_v2"} <= set(operators) else "legacy_v1" if "ts_corr" in operators else "pearson_v2" if "ts_corr_v2" in operators else "not_used",
        "canonical": canonical,
        "operators": sorted(set(operators)),
        "fields": sorted(set(fields)),
        "windows": sorted(set(windows)),
        "required_history": required_history(expression),
        "complexity": sum(1 for _ in ast.walk(tree)),
        "latex": expression_to_latex(expression),
    }
