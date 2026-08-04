"""因子表达式 DSL: Python 语法子集 -> polars 多阶段流水线.

安全性: 仅白名单 AST 节点/算子/字段; 所有时序算子仅向后看, 结构上无前视.
实现要点: polars 不支持嵌套窗口表达式 (over 内含 over 会静默产生 null),
因此每个窗口算子的结果都物化为临时列, 保证任何阶段内 over 均为单层。
"""

import ast
import hashlib

import polars as pl

from ..config import DSL_FIELDS

_BY_CODE = {"partition_by": "ts_code", "order_by": "trade_date"}


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
    if not (isinstance(node_w, ast.Constant) and isinstance(node_w.value, int) and 1 <= node_w.value <= 250):
        raise ValueError("窗口参数必须是 1..250 的整数字面量")
    return node_w.value


OPERATORS_DOC = {
    "ts_mean(x, w)": "w 日滚动均值",
    "ts_std(x, w)": "w 日滚动标准差",
    "ts_sum(x, w)": "w 日滚动求和",
    "ts_min(x, w)": "w 日滚动最小",
    "ts_max(x, w)": "w 日滚动最大",
    "ts_rank(x, w)": "当前值在过去 w 日中的分位",
    "ts_delta(x, w)": "x - delay(x, w)",
    "ts_corr(x, y, w)": "w 日滚动相关系数",
    "delay(x, d)": "滞后 d 日",
    "rank(x)": "当日截面百分位排名",
    "zscore(x)": "当日截面 zscore",
    "winsor(x)": "当日截面 2.5 倍标准差截尾",
    "log(x)": "log(|x|+1e-9)",
    "abs(x)": "绝对值",
    "sign(x)": "符号",
}


def _percentile_of_last(s: pl.Series) -> float | None:
    last = s[-1]
    if last is None:
        return None
    return float((s <= last).sum()) / len(s)


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
    def __init__(self) -> None:
        self.stages: list[tuple[str, pl.Expr]] = []

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
            if node.id not in DSL_FIELDS:
                raise ValueError(f"未知字段: {node.id} (白名单: {DSL_FIELDS})")
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
            return self._mat(_ts(x.rolling_map(_percentile_of_last, w, min_samples=w)))
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


def parse(expression: str) -> FactorPipeline:
    """解析 DSL 表达式为 FactorPipeline; 非法即抛 ValueError."""
    if len(expression) > 500:
        raise ValueError("表达式过长")
    tree = ast.parse(expression, mode="eval")
    _degenerate_check(tree.body)
    b = _Builder()
    final = b.build(tree.body)
    return FactorPipeline(b.stages, final)


def normalize_hash(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()[:16]


def validate(expression: str) -> str | None:
    try:
        parse(expression)
        return None
    except (ValueError, SyntaxError) as e:
        return str(e)
