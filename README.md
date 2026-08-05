# USStockFactorFactory

7×24 双层嵌套优化 (bi-level) LLM 因子挖掘工厂。外层 Meta-Optimizer 进化「挖掘器配置 (HarnessSpec)」，
内层 Miner 在该配置下进化「因子表达式」；四级时间隔离防止过拟合泄漏。设计蓝图见 [DESIGN.md](DESIGN.md)。

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
| 因子库 | 经典种子因子 + 自动挖掘因子; 详情抽屉含四层 era IC 步进图与分层指标; 手动评估录入 |
| 回测 | 手动回测 (多空/纯多头、成本、方向), 费后净值曲线 + 绩效统计 + 历史记录 |
| 设置 | OpenAI/Anthropic 格式大模型接入 (多提供商, 内/外层分别指定)、引擎参数 |

## 架构

- **外层**: 白名单内进化 HarnessSpec (n_drafts / improve_bias / temperature / 入库门槛...)，
  ε-占优才接受，在位者定期重测 (noise band)。
- **内层**: LLM (未配置时随机基线) 起草/改进 DSL 因子表达式 → AST 白名单编译为 polars
  流水线 → 评估 RankIC/ICIR/era 一致性/换手。
- **数据隔离**: INNER_PUBLIC(2010-2019, 进提示词) / META_TRAIN(2020-2022, gate 仅存库) /
  META_HOLDOUT(2023-2024) / FACTOR_VAULT(2025+, 永不进循环)。
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

> 数据免责: 面板为非 PIT 当前成分并集 (幸存者偏差)，`production_eligible=false`，仅供研究。
