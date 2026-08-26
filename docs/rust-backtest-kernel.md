# Rust 回测内核镜像

8765 继续是 Python `step_event_v2` 权威服务；8764 是隔离的 Rust 影子镜像。两者共享研究数据库，但8764使用独立的回测产物目录，且不会自动启动研究任务。

## 当前执行协议

- Rust ABI：`factorfactory-rust-backtest/0.2.0` / ABI v2。
- Rust协议：`step_event_v3_rust_shadow`。
- 数据输入：Polars完成DSL与因子物化后，以连续列式数组通过同步FFI进入Rust；热循环内不创建Python字典。
- 信号与执行：t日收盘生成目标，t+1原始开盘成交。
- 已覆盖：A股纯多、美股纯多、美股多空、等权、固定/线性/平方根冲击、历史费率、ADV容量、A股T+1、现金/保证金、借券费、公司行动持仓连续、开盘杠杆修复。
- 自动回退：逆波动率/ATR定仓、carry订单、逐股止损止盈、组合熔断与期末强平仍走Python。

## 可信门槛

Rust不能只比较最终收益。每次影子运行依次检查：

1. 成交笔数及成交日、证券、买卖方向完全一致；
2. 每笔数量、成交价、费用、现金和成交后持仓绝对误差不超过 `1e-5`；
3. 每日现金、市值、NLV、收益、换手和融资费绝对误差不超过 `1e-5`；
4. 原Python交割单完整性检查仍须全部通过。

任何一项失败都会标记 `python_fallback`，返回Python权威结果并保留首个分歧点，不会静默使用Rust结果。

## 运行

```bash
./scripts/build_rust_backtest_kernel.sh
./service-8764.sh restart
./service-8764.sh status
```

复现实盘面板对齐与性能测试：

```bash
.venv/bin/python scripts/benchmark_rust_backtest.py \
  --start 2023-01-01 --end 2024-12-31 --universe 100
```

API能力与后端状态：

```bash
curl -fsS http://127.0.0.1:8764/api/backtest/capabilities
```

## 2026-08-24基线

表达式 `rank(returns(close, 20))`，Top100，2023-2024，每5日调仓：

| 市场/模式 | 日数 | 成交 | 逐笔身份差异 | 每日NLV最大误差 | Rust事件核心加速 |
|---|---:|---:|---:|---:|---:|
| A股纯多 | 484 | 1,332 | 0 | 0 | 约52x |
| 美股纯多 | 502 | 1,451 | 0 | 0 | 约47-56x |
| 美股多空 | 502 | 5,137 | 0 | 0 | 约59x |

冷面板端到端收益会被DSL与因子物化主导；当前观测约1.05-1.13x。面板缓存命中时，美股多空端到端预测约2.8-3.4x。因此后续优化重点应是因子物化缓存与列式计划复用，而不是降低账本检查。
