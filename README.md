# FactorFactory

支持 A 股与美股并行任务的 7×24 双层嵌套优化 (bi-level) LLM 因子挖掘工厂。外层 Meta-Optimizer 进化「挖掘器配置 (HarnessSpec)」，
内层 Miner 在该配置下进化「因子表达式」；Evaluation Protocol V4 把搜索发现分、实盘排序分与最终校准分离。
设计蓝图见 [DESIGN.md](DESIGN.md)，事件回测的冻结口径见
[docs/BACKTEST_PROTOCOL_V1.md](docs/BACKTEST_PROTOCOL_V1.md)。

## 快速开始

```bash
createdb factor_factory      # 首次
./service.sh start           # 建 venv、锁定依赖、后台启动唯一实例
./service.sh status          # PID、端口与 readiness
./service.sh logs            # 最近服务日志；加 -f 持续跟踪
./service.sh restart
./service.sh stop
```

访问 http://localhost:10010。`service.sh` 使用当前 macOS 用户的 launchd 会话托管并防止双实例；
`run.sh` 仍可用于前台开发。依赖文件未变化时不会重复安装，敏感环境变量可写入已忽略的 `.env`，
无需出现在进程命令行。

## 页面

| 页面 | 内容 |
|---|---|
| 总览 | 引擎启停、外层 meta-score 步进图 (候选 vs 在位)、在位 Miner 配置、实时日志 |
| 研发树 | Miner 版本演化表 + 内层搜索树 (draft/improve 血缘, 可缩放, 点击看表达式) |
| 因子库 | V4 四层审计、费后实盘排序及 Vault 校准诊断、F1–F5 生命周期、结构相似度分组、Web LaTeX |
| 选股器 | 单次 Polars 懒执行的多因子或直接 DSL 截面选股；历史窗裁剪、结果缓存与逐股因子归因 |
| 回测 | `t` 收盘信号 → `t+1` 原始开盘成交的步进事件引擎、逐日状态、事件流、交割单与完整性门 |
| 诊断 | 滚动窗口 P50/P95/P99、显式 SLO、5xx 与请求 ID、进程/事件循环、数据库连接池、面板 schema/DSL 契约、采集开销、worker 心跳与跨重启事故日志 |
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
- **V4 实盘排序**: 0–100 分以 HOLDOUT 费后收益/Sharpe 下置信界、收益 HAC、多重检验门槛、
  成本盈亏平衡、压力成本、跨 era 盈利率、泛化衰减和容量为核心；Vault 数值不进入公式，
  仅用于检验冻结排序与后续费后结果的 Spearman、Top 组盈利率和分组单调性。
- **交互性能**: 页面使用 KeepAlive、GET 去重/短缓存和非重载任务切换；列表 API 只返回指标摘要，
  worker 状态不再重复携带日志，元信息接口也不触发冷面板全量加载。
- **可观测性**: `/api/health/live` 提供轻量存活检查，`/api/health/ready` 验证数据库与任务面板，
  `/api/observability` 返回脱敏工程快照，`/api/metrics` 暴露低基数 Prometheus 指标。慢采集层采用
  single-flight TTL 缓存，避免诊断轮询反过来拖慢研究；严重错误写入有界滚动 JSONL，跨重启保留。
- **运行安全**: 默认只监听 `127.0.0.1`，API 响应禁止缓存并附带基础浏览器安全头；前端 CDN 版本精确锁定
  且启用 SRI。非本机监听必须显式设置不安全确认变量，防止把无认证控制面意外暴露到局域网。
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
| `FF_HOST` | 127.0.0.1（仅本机） |
| `FF_PORT` | 10010 |
| `FF_ALLOW_REMOTE_UNAUTHENTICATED` | 空；仅非本机监听且确认隔离网络时设为 1 |
| `FF_DATABASE_URL` | postgresql+asyncpg://jiangjingzhe@localhost:5432/factor_factory |
| `FF_PANEL_GLOB` | MultiFactorUS yfinance 研究面板 parquet 路径 |

> 数据免责: 当前 V4 按用户要求不把 PIT 纳入评分，但面板仍标记为非 PIT 当前成分并集。
> 所有等级都属于 `NON_PIT_RESEARCH`，不等同于生产或实盘批准。
