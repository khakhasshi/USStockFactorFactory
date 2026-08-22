"""因子表达式 DSL: Python 语法子集 -> polars 多阶段流水线.

安全性: 仅白名单 AST 节点/算子/字段; 所有时序算子仅向后看, 结构上无前视.
实现要点: polars 不支持嵌套窗口表达式 (over 内含 over 会静默产生 null),
因此每个窗口算子的结果都物化为临时列, 保证任何阶段内 over 均为单层。
"""

import ast
import hashlib
import math
import re

import polars as pl

from ..config import DSL_FIELDS

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
    "ts_corr(x, y, w)": "w 日滚动相关系数",
    "delay(x, d)": "滞后 d 日",
    "rank(x)": "当日截面百分位排名",
    "zscore(x)": "当日截面 zscore",
    "winsor(x)": "当日截面 2.5 倍标准差截尾",
    "winsor_mad(x, threshold)": "当日截面 threshold 倍 MAD 截尾",
    "log(x)": "log(|x|+1e-9)",
    "abs(x)": "绝对值",
    "sign(x)": "符号",
}


class FactorPipeline:
    """有序临时列阶段 + 最终表达式; apply 后临时列被丢弃."""

    def __init__(self, stages: list[tuple[str, pl.Expr]], final: pl.Expr) -> None:
        self.stages = stages
        self.final = final

    def apply(self, lf: pl.LazyFrame, alias: str = "factor") -> pl.LazyFrame:
        for name, expr in self.stages:
            lf = lf.with_columns(expr.alias(name))
        lf = lf.with_columns(self.final.alias(alias))
        tmp = [n for n, _ in self.stages]
        return lf.drop(tmp) if tmp else lf


class _Builder:
    def __init__(self, fields: list[str] | None = None) -> None:
        self.stages: list[tuple[str, pl.Expr]] = []
        self.fields = fields or DSL_FIELDS

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
            return self._call(node)
        raise ValueError(f"非法语法节点: {type(node).__name__}")

    def _call(self, node: ast.Call) -> pl.Expr:
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise ValueError("非法调用形式")
        fn = node.func.id
        a = node.args

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
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Sub, ast.Div)):
            if ast.dump(n.left) == ast.dump(n.right):
                raise ValueError("退化表达式: x-x / x/x 恒为常数")
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "ts_corr":
            if len(n.args) >= 2 and ast.dump(n.args[0]) == ast.dump(n.args[1]):
                raise ValueError("退化表达式: ts_corr(x, x, w) 恒为 1")


def parse(expression: str, fields: list[str] | None = None) -> FactorPipeline:
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
    b = _Builder(fields)
    final = b.build(tree.body)
    return FactorPipeline(b.stages, final)


def normalize_hash(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()[:16]


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
    return min(1000, max(1, _history_for_node(tree.body)))


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
        "canonical": canonical,
        "operators": sorted(set(operators)),
        "fields": sorted(set(fields)),
        "windows": sorted(set(windows)),
        "required_history": required_history(expression),
        "complexity": sum(1 for _ in ast.walk(tree)),
        "latex": expression_to_latex(expression),
    }
