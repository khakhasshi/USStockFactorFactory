# USStockFactorFactory

7×24 双层嵌套优化 (bi-level) LLM 因子挖掘工厂。外层 Meta-Optimizer 进化「挖掘器配置 (HarnessSpec)」，
内层 Miner 在该配置下进化「因子表达式」；Evaluation Protocol V3 把搜索评分与实战准入分离。
设计蓝图见 [DESIGN.md](DESIGN.md)，事件回测的冻结口径见
[docs/BACKTEST_PROTOCOL_V1.md](docs/BACKTEST_PROTOCOL_V1.md)。

## 快速开始

```bash
createdb factor_factory      # 首次
./run.sh                     # 建 venv、装依赖、启动服务
```

访问 http://localhost:10010 (可用 `FF_PORT=xxxx ./run.sh` 换端口)。

## 页面

| 页面 | 内容 |
|---|---|
| 总览 | 引擎启停、外层 meta-score 步进图 (候选 vs 在位)、在位 Miner 配置、实时日志 |
| 研发树 | Miner 版本演化表 + 内层搜索树 (draft/improve 血缘, 可缩放, 点击看表达式) |
| 因子库 | V3 四层审计、F1–F5 生命周期、结构相似度快速分组/近邻搜索、DSL 与 Web LaTeX 切换 |
| 选股器 | 单次 Polars 懒执行的多因子或直接 DSL 截面选股；历史窗裁剪、结果缓存与逐股因子归因 |
| 回测 | `t` 收盘信号 → `t+1` 原始开盘成交的步进事件引擎、逐日状态、事件流、交割单与完整性门 |
| 设置 | A股/美股研究任务、纯多头/多空、冻结信号方向、成本/容量/OOS 门槛及模型接入 |

## 架构

- **外层**: 白名单内进化 HarnessSpec (n_drafts / improve_bias / temperature / 入库门槛...)，
  ε-占优才接受，在位者定期重测 (noise band)。
- **内层**: LLM (未配置时随机基线) 起草/改进 DSL 因子表达式 → 任务市场专属 AST 白名单
  → 评估 RankIC/HAC、真实组合换手、费后收益、市场 Beta、分层单调性与成本压力。
- **数据隔离**: INNER_PUBLIC(2010-2019, 进提示词) / META_TRAIN(2020-2022, gate 仅存库) /
  META_HOLDOUT(2023-2024) / FACTOR_VAULT(2025+, 仅显式完整审计读取，永不进循环)。
- **生命周期**: F1 discovery → F2 research-pass → F3 OOS-pass → F4 paper-candidate →
  F5 live-candidate-non-pit。F5 仍是 `NON_PIT_RESEARCH`，不是生产批准。
- **交互性能**: 页面使用 KeepAlive、GET 去重/短缓存和非重载任务切换；列表 API 只返回指标摘要，
  worker 状态不再重复携带日志，元信息接口也不触发冷面板全量加载。
- **事件回测**: A股使用万2免5及历史印花税/过户费，美股使用 IBKR Pro Fixed；
  CSV/Parquet 交割单、事件账本、逐日账本与 SHA-256 manifest 来自同一个状态引擎。
- **因子资产索引**: 规范化 AST、SimHash LSH 与加权 Jaccard 先快速召回再精排，
  不读取 HOLDOUT/VAULT 收益来决定结构相似度。
- **技术栈**: FastAPI + SQLAlchemy(asyncpg) + polars / Vue3 + ECharts (CDN, 无构建) / PostgreSQL。

## DSL 算子

`ts_mean/std/sum/min/max/rank/delta/corr`, `delay`, `rank/zscore/winsor` (截面), `log/abs/sign`,
四则运算; 字段 `open high low close vol amount` (前复权研究口径)。

## 环境变量

| 变量 | 默认 |
|---|---|
| `FF_PORT` | 10010 |
| `FF_DATABASE_URL` | postgresql+asyncpg://jiangjingzhe@localhost:5432/factor_factory |
| `FF_PANEL_GLOB` | MultiFactorUS yfinance 研究面板 parquet 路径 |

> 数据免责: 当前 V3 按用户要求不把 PIT 纳入评分，但面板仍标记为非 PIT 当前成分并集。
> 所有等级都属于 `NON_PIT_RESEARCH`，不等同于生产或实盘批准。
