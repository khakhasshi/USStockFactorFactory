# FactorFactory A 股与美股数据：Agent 快速接入提示词

更新时间：2026-08-22  
适用项目：`Portfolios_USStockFactorFactory` 及其内部研究 demo 分支  
用途：把下方“可复制提示词”完整交给新的研究 Agent，使其在不猜测数据口径、不越过数据隔离边界的前提下快速开始因子研究、DSL 编写、验证或回测。

## 可直接复制给 Agent 的提示词

```text
你正在为本地 FactorFactory 做 A 股和美股的内部因子研究。以下数据合同与隔离规则是硬约束。开始工作前先选择 market=ashare 或 market=us；不要把两个市场的字段白名单混用。

一、项目与数据位置

生产项目：
/Users/jiangjingzhe/Finance_Data_Center/releases/source_roots/Portfolios_USStockFactorFactory

内部研究 demo 工作树（如果任务明确要求研究 demo）：
/Users/jiangjingzhe/Finance_Data_Center/releases/source_roots/Portfolios_USStockFactorFactory-demo-20010

美股 Parquet 年度分区：
/Users/jiangjingzhe/Portfolios/MultiFactorUS/data_yfinance_research/processed/daily_panel/trade_year=*/data_0.parquet

A 股 Parquet 年度分区：
/Users/jiangjingzhe/Portfolios/MultiFactorAshare/data/trade_year=*/data_0.parquet

两个面板当前各有 17 个年度 Parquet 分区。使用 Polars LazyFrame 和 hive_partitioning=True 扫描；先做列裁剪和日期过滤，不要无条件把整个面板复制进 pandas。

当前任务配置中的 panel_glob 优先于任何默认路径。若服务可访问，可以查看 /api/experiments 中目标任务的 panel_glob；/api/meta 只反映当前激活任务，不能据此推断另一个市场。

二、2026-08-22 核验的数据快照

美股：
- 时间范围：2010-06-01 至 2026-08-21
- 行数：8,755,516
- 证券数：2,895
- 证券代码列：ts_code
- 原始面板体积约 655 MB

A 股：
- 时间范围：2010-01-04 至 2026-08-21
- 行数：9,573,747
- 证券数：3,196
- 证券代码列：ts_code
- 原始数据目录约 1.7 GB；其中还可能包含非面板文件，因此不是纯 Parquet 精确体积

日期、行数、证券数会随数据更新变化。报告中必须写明实际查询到的 date_min、date_max、row_count、security_count 和 panel_glob，不能盲目复用上述快照。

三、因子 DSL 真正允许使用的字段

美股只允许 6 个字段：
open, high, low, close, vol, amount

A 股允许 20 个字段：
open, high, low, close, vol, amount,
pe_ttm, pb, ps_ttm, dv_ttm,
total_mv, circ_mv,
turnover_rate, volume_ratio,
net_mf_amount,
buy_lg_amount, sell_lg_amount,
buy_elg_amount, sell_elg_amount,
float_share

字段语义：
- open/high/low/close：前复权价格，来源为 panel_adjusted_ohlc，单位为价格。
- vol：原始报告成交股数，未随前复权价格缩放。
- amount（A 股）：原始货币成交额。
- amount（美股）：raw_close × volume 的估算代理，不是交易所直接报告的精确成交额；面板中的 amount_source 和 amount_is_estimated 可用于审计，但不能写进 DSL。
- pe_ttm/pb/ps_ttm/dv_ttm：A 股数据提供商的日频快照估值比率。
- total_mv/circ_mv：A 股原始货币口径总市值/流通市值。
- turnover_rate/volume_ratio：A 股数据提供商日频快照比率。
- net_mf_amount、buy/sell_lg_amount、buy/sell_elg_amount：A 股原始货币口径资金流。
- float_share：A 股原始报告流通股本。

不要使用字段名 ps；合法字段是 ps_ttm。不要把 A 股特有字段用于美股表达式。

四、原始列、内部派生列与禁用规则

Parquet 里存在的列不等于 DSL 可用字段。以下列只能用于数据审计、标签生成、交易可行性检查或引擎内部计算，禁止作为因子输入：

- 标识列：trade_date, ts_code, name
- 原始价格与复权审计：raw_open, raw_high, raw_low, raw_close, raw_pre_close, adj_close, adjustment_factor, adj_factor 等
- 数据质量与来源：knowledge_time, source_batch, revision_id, amount_source, amount_is_estimated, history_observations 及各类 is_* 质量标记
- 交易可行性代理：can_buy_open_proxy, can_sell_open_proxy, up_limit, down_limit
- 引擎未来标签：fwd_1, fwd_5, fwd_10, fwd_20
- 引擎分层/截面辅助：amt60, univ_rank, era, layer
- 其他未列入对应市场 DSL 白名单的原始列

特别禁止使用 fwd_*、负数 delay、未来 shift、未来日期信息或任何从评级结果反推表达式的做法。fwd_* 是引擎生成的监督/评价标签，不是特征。

五、数据质量过滤和交易标签口径

PanelStore 加载时会按存在性使用以下质量过滤：
- 两市场：is_tradable_observation、is_valid_ohlc
- 美股额外：is_security_identity_consistent

当前快照中质量过滤后的行数与上述原始统计相同，但每次数据更新后仍应重新验证。

引擎前向收益定义为：t 日形成信号，t+1 日开盘成交，在 t+1+h 日开盘平仓；价格采用前复权 open。原生评价 horizon 为 1、5、10、20 个交易日。

研究股票池使用 rolling_60d_amount_rank_v1：先计算每只证券 60 日平均 amount（至少 20 个样本），再按交易日做流动性降序排名 univ_rank，并取任务指定的 universe_n。不要把未来成交额用于当日股票池。

六、时间隔离与 Agent 可见性

A 股：
- INNER_PUBLIC：2010-01-01 至 2019-12-31（实际从 2010-01-04 开始）
- META_TRAIN：2020-01-01 至 2022-12-31
- META_HOLDOUT：2023-01-01 至 2024-12-31
- FACTOR_VAULT：2025-01-01 至 2026-08-04

美股：
- INNER_PUBLIC：2010-06-01 至 2019-12-31
- META_TRAIN：2020-01-01 至 2022-12-31
- META_HOLDOUT：2023-01-01 至 2024-12-31
- FACTOR_VAULT：2025-01-01 至 2026-08-04

发现/优化循环只能从 INNER_PUBLIC 与 META_TRAIN 获得保守聚合反馈。方向必须在训练安全层选择并冻结。不得把 META_HOLDOUT、FACTOR_VAULT 或冻结评级的数值结果放回候选生成提示词。

显式 FULL_AUDIT_V4 才能读取四个声明层。冻结评级协议为 V4.3：使用已冻结方向，从 2020-01-01 计算到面板最新日期（当前为 2026-08-21），并可包含 2026-08-04 之后的 post-vault extension。该评级跨越多个研究层，因此是全历史诊断评级，不是独立样本外证据；visible_to_research_llms=false。

七、PIT 与可生产性边界

两个面板均标记为：
- pit_quality=non_pit_current_constituents
- production_eligible=false
- 结果标签=NON_PIT_RESEARCH

当前项目约定：快速因子研究不因缺少 PIT 单独卡住，也不把 PIT 作为入库硬门槛。但是必须如实披露 current-constituent/non-PIT 边界，不得声称结果已达到生产可交易标准。未来数据泄露、错误时间对齐、错误复权、证券身份错配仍是硬性失败，不因“不考虑 PIT”而放宽。

A 股估值、股本和资金流字段是 provider snapshot；不能把它们描述成经过严格 point-in-time 版本化的财务数据。

八、调整口径与量纲硬规则

open/high/low/close 是前复权价格，而 vol/amount 是原始口径。禁止把绝对前复权价格尺度直接和原始成交额混合，例如 (high-low)/amount。先把价格变化归一化成无量纲收益或振幅比例，再与 amount/vol 组合。

可接受示例：
rank(ts_mean((high-low)/(abs(close)+1e-9), 20))
rank(ts_corr(returns(close, 1), log(amount), 20))

不接受示例：
rank(ts_mean((high-low)/amount, 20))
fwd_5
rank(ps)

九、DSL 语法

允许普通算术：+, -, *, /，允许负号和有限数值常量。

允许算子：
- ts_mean(x,w), ts_std(x,w), ts_sum(x,w), ts_min(x,w), ts_max(x,w)
- ts_rank(x,w), ts_delta(x,w), returns(x,w)
- ts_corr(x,y,w), delay(x,d)
- rank(x), zscore(x), winsor(x), winsor_mad(x,threshold)
- log(x), abs(x), sign(x)

所有时序运算按 ts_code 分组并按 trade_date 排序，只向后看；rank/zscore/winsor 为当日截面操作。新研究候选的窗口使用 1..250 的整数字面量。解析器仅为兼容旧库额外接受 251/252，不要把它扩大为新搜索空间。winsor_mad threshold 必须在 (0,20]。表达式最多 4000 字符、1000 个 AST 节点。禁止 x-x、x/x、ts_corr(x,x,w) 等退化表达式。

十、研究输出的最低审计信息

每个研究结论至少记录：
- market、panel_glob、数据最新日、实际研究日期区间
- expression、规范化 hash、字段、算子、最大所需历史长度
- mechanism_family、经济假设、预期失效条件
- direction（+1=高值侧做多，-1=低值侧做多）及方向冻结依据
- horizon、portfolio_mode、universe_n、调仓/成交假设
- RankIC、IC、ICIR、HAC t/p、覆盖率、单调性、换手
- 费后收益、Sharpe、最大回撤、成本压力、容量与 long/short 腿归因
- INNER_PUBLIC/META_TRAIN/META_HOLDOUT/FACTOR_VAULT 分层结果
- 多重检验负担、重复因子/重复收益来源检查
- NON_PIT_RESEARCH 与 production_eligible=false 声明

不要只凭总榜分、单一 Sharpe 或冻结全历史评级宣布因子有效。若存在未来函数、时间对齐错误、复权错配、字段越权或证券身份问题，直接判定无效；若只是 PIT 不足，则继续快速研究但保留醒目标注。

十一、开始任务时必须先执行的检查

1. 确认当前 checkout、分支、脏文件和目标任务的 market/panel_glob。
2. 用 LazyFrame 查询实际 schema、date_min/date_max、行数和证券数。
3. 将原始列与对应市场 DSL 白名单求交集；绝不因为 Parquet 有某列就擅自开放。
4. 确认研究阶段允许访问的 layer，不读取密封层结果来生成候选。
5. 在运行大规模回测前，先用 DSL validate/semantic audit 检查字段、窗口、未来函数、量纲和所需历史长度。
6. 任何结果都附带数据快照和协议版本，确保可以复现。

如果你的任务是提出因子表达式，只输出本市场白名单内、符合上述规则的最小可证伪 DSL；同时说明机制、方向假设、所需历史、失效条件。不要用叙事代替实证评价。
```

## 数据快照刷新命令

在生产项目根目录运行：

```bash
.venv/bin/python - <<'PY'
import polars as pl

panels = {
    "us": "/Users/jiangjingzhe/Portfolios/MultiFactorUS/data_yfinance_research/processed/daily_panel/trade_year=*/data_0.parquet",
    "ashare": "/Users/jiangjingzhe/Portfolios/MultiFactorAshare/data/trade_year=*/data_0.parquet",
}

for market, path in panels.items():
    lf = pl.scan_parquet(path, hive_partitioning=True)
    schema = lf.collect_schema()
    stats = lf.select(
        pl.len().alias("rows"),
        pl.col("trade_date").min().alias("date_min"),
        pl.col("trade_date").max().alias("date_max"),
        pl.col("ts_code").n_unique().alias("securities"),
    ).collect()
    print(f"\n[{market}] {path}")
    print(stats)
    print(schema)
PY
```

如果要检查正在运行的服务，应先确认端口与目标任务，再运行：

```bash
BASE_URL=http://127.0.0.1:10010
curl -fsS "$BASE_URL/api/experiments"
curl -fsS "$BASE_URL/api/meta"
```

`/api/meta` 只代表当前激活实验。统一端口同时承载 A 股和美股任务时，必须从 `/api/experiments` 的任务配置确认各自 `market` 和 `panel_glob`。

## 权威实现位置

- 面板默认路径、字段白名单、时间分层和评分配置：`backend/app/config.py`
- 面板加载、质量过滤、前向标签、流动性股票池和 layer 构造：`backend/app/data/panel.py`
- DSL 语法、算子、窗口和复杂度限制：`backend/app/dsl/engine.py`
- 字段来源、调整口径与量纲审计：`backend/app/factors/semantics.py`
- 研究可见层、FULL_AUDIT_V4 与冻结评级：`backend/app/eval/harness.py`

发生文档与当前任务配置或实现冲突时，以目标任务持久化的 `panel_glob` 和当前 checkout 的上述代码为准，并在报告中记录差异。
