"""随机基线: 复刻内层 draft/improve 爬山流程, 但用随机生成器替代 LLM.

只读评估, 不写数据库, 结果落盘 JSON. 用法:
  ../.venv/bin/python -m scripts.random_baseline
"""

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.eval.harness import evaluate  # noqa: E402
from app.miner.agent import mutate_expression, random_expression  # noqa: E402

# 与 LLM 已产出节点相同的任务配比
PLAN = [
    ({"name": "T1_liquid500_5d", "universe_n": 500, "horizon": 5}, 109),
    ({"name": "T2_mid1500_10d", "universe_n": 1500, "horizon": 10}, 83),
    ({"name": "T3_liquid500_20d", "universe_n": 500, "horizon": 20}, 78),
]
IMPROVE_BIAS = 0.6  # 与内层 spec 常见值一致
OUT = Path(__file__).resolve().parent / "random_baseline_results.json"


def main() -> None:
    random.seed(20260804)
    results: list[dict] = []
    t0 = time.time()
    total = sum(n for _, n in PLAN)
    done = 0
    for task, rounds in PLAN:
        best_expr: str | None = None
        best_score = -1.0
        seen: set[str] = set()
        for _ in range(rounds):
            # draft/improve 爬山, 与内层回退逻辑一致
            for _retry in range(10):
                if best_expr is not None and random.random() < IMPROVE_BIAS:
                    expr, op = mutate_expression(best_expr), "improve"
                else:
                    expr, op = random_expression(), "draft"
                if expr not in seen:
                    break
            seen.add(expr)
            row = {"task": task["name"], "op": op, "expression": expr}
            try:
                m = evaluate(expr, task["universe_n"], task["horizon"])
                pub = m["public"]
                row.update(score=pub["score"], icir=pub["icir"], ic_mean=pub["ic_mean"],
                           era_consistency=pub["era_consistency"], turnover=pub["turnover"])
                if pub["score"] is not None and pub["score"] > best_score:
                    best_score, best_expr = pub["score"], expr
            except Exception as e:  # noqa: BLE001
                row.update(score=None, error=str(e)[:120])
            results.append(row)
            done += 1
            if done % 10 == 0 or done == total:
                el = time.time() - t0
                print(f"[{done}/{total}] {task['name']} best={best_score:.4f} elapsed={el:.0f}s", flush=True)
                OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"完成, 写入 {OUT}")


if __name__ == "__main__":
    main()
