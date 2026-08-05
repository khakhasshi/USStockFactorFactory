# FactorFactory

支持 A 股与美股并行任务的 7×24 双层嵌套优化 (bi-level) LLM 因子挖掘工厂。外层 Meta-Optimizer 进化白名单约束的
`MinerTemplate`（提示词、搜索策略、反馈上下文、示例优先级与 DSL 结构），内层 Miner 在该模板下进化因子表达式；
Evaluation Protocol V4.2 把连续学习分、硬准入分、实盘排序分与最终校准分离；
每个新候选在训练安全层同时评价正反方向、按双向试验数惩罚后冻结方向，
失败候选也保留可比较的严重程度，避免双层 LLM 面对一片 `0.000` 无法归因。
设计蓝图见 [DESIGN.md](DESIGN.md)，事件回测的冻结口径见
[docs/BACKTEST_PROTOCOL_V1.md](docs/BACKTEST_PROTOCOL_V1.md)，跨任务全因子榜单口径见
[docs/FACTOR_LEADERBOARD_PROTOCOL_V1.md](docs/FACTOR_LEADERBOARD_PROTOCOL_V1.md)。

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

冻结任务 1–8 的历史节点与因子并用多进程回放 A 股榜单：

```bash
.venv/bin/python backend/scripts/factor_leaderboard.py \
  --max-experiment-id 8 --max-node-id 2208 --max-factor-id 1522 \
  --workers 2 --threads-per-worker 4
```

运行记录逐条落入 `var/reports`，中断后以相同 `--output-dir` 重跑即可续算。
完成后可从冻结结果生成无网络依赖的单文件交互报表：

```bash
.venv/bin/python backend/scripts/render_factor_leaderboard_html.py \
  --report-dir var/reports/ashare-factor-leaderboard-YYYYMMDD-HHMMSS
```

## 页面

| 页面 | 内容 |
|---|---|
| 总览 | 引擎启停、外层 meta-score 步进图 (候选 vs 在位)、在位 Miner 配置、实时日志 |
| 研发树 | Miner 版本演化表 + 内层搜索树 (draft/improve 血缘, 可缩放, 点击看表达式) |
| 因子库 | V4 四层审计、费后实盘排序及 Vault 校准诊断、F1–F5 生命周期、结构相似度分组、Web LaTeX |
| 选股器 | 单次 Polars 懒执行的多因子或直接 DSL 截面选股；历史窗裁剪、结果缓存与逐股因子归因 |
| 回测 | `t` 收盘信号 → `t+1` 原始开盘成交的步进事件引擎、逐日状态、事件流、交割单与完整性门 |
| 诊断 | 滚动窗口 P50/P95/P99、显式 SLO、5xx 与请求 ID、进程/事件循环、数据库连接池、面板 schema/DSL 契约、双层 LLM 调用/反馈血缘、协议隔离、worker 心跳与跨重启事故日志 |
| 设置 | A股/美股研究任务、纯多头/多空、冻结信号方向、成本/容量/OOS 门槛及模型接入 |

## 架构

- **外层**: 白名单内进化 `MinerTemplate`；先读取同协议的跨任务/多种子报告与上轮结果反思，
  再做最小可归因改动。候选与在位者使用同一冻结历史基线，按多种子单边统计门接受或拒绝。
- **内层**: LLM (未配置时随机基线) 起草/改进 DSL 因子表达式 → 任务市场专属 AST 白名单
  → 评估 RankIC/HAC、真实组合换手、费后收益、市场 Beta、分层单调性与成本压力。下一轮会收到
  保守有效指标、评分组件、明确失败原因、改进目标与先前反思，不再只看到 score/ICIR。
- **数据隔离**: INNER_PUBLIC + META_TRAIN 只以预声明的保守聚合反馈进入双层循环；原始明细不进 prompt。
  META_HOLDOUT(2023-2024) / FACTOR_VAULT(2025+) 仅由显式完整审计读取，永不进入 Miner 或外层反馈。
- **协议与种子隔离**: 历史协议数据只读保留；当前上下文、meta-score 与外层比较只查询同一
  `evaluation_protocol`。每个种子看到同一冻结历史基线和本种子增量，不会从其他种子继续学习。
- **单一研究引擎**: UI/API 的新运行只允许 V2；V1 代码与既有结果仅作历史审计，不能再写入当前协议。
- **反思闭环**: 内层提案保存所针对的失败和预期改善；外层在决策后保存
  `supported/refuted/inconclusive`、证据、经验、避免模式、下一实验与停止条件，供下一步读取。
- **生命周期**: F1 discovery → F2 research-pass → F3 OOS-pass → F4 paper-candidate →
  F5 live-candidate-non-pit。F5 仍是 `NON_PIT_RESEARCH`，不是生产批准。
- **V4.2 方向与实盘排序**: 新候选在训练安全层同时评价 +1/-1，按两次试验计数后冻结方向；
  0–100 分以 HOLDOUT 费后收益/Sharpe 下置信界、收益 HAC、多重检验门槛、
  成本盈亏平衡、压力成本、跨 era 盈利率、泛化衰减和容量为核心；Vault 数值不进入公式，
  仅用于检验冻结排序与后续费后结果的 Spearman、Top 组盈利率和分组单调性。
- **交互性能**: 页面使用 KeepAlive、GET 去重/短缓存和非重载任务切换；列表 API 只返回指标摘要，
  worker 状态不再重复携带日志，元信息接口也不触发冷面板全量加载。
- **可观测性**: `/api/health/live` 提供轻量存活检查，`/api/health/ready` 验证数据库与任务面板，
  `/api/observability` 返回脱敏工程快照，`/api/metrics` 暴露低基数 Prometheus 指标。慢采集层采用
  single-flight TTL 缓存，避免诊断轮询反过来拖慢研究；严重错误写入有界滚动 JSONL，跨重启保留。
  `/api/llm/audits` 追加保存脱敏 prompt/response、prompt hash、反馈 fingerprint、模型、阶段、延迟和错误，
  可证明每次内外层调用究竟看到了哪一版反馈；provider 密钥从不落入审计记录。
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
