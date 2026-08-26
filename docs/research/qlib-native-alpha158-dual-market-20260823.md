# Qlib Native / Alpha158 双市场审计

日期：2026-08-23  
结论标签：`NON_PIT_RESEARCH`、`NO_PROMOTION`

## 本次吸收的能力

- 固定 Microsoft Qlib 上游提交 `79633dd9506ea689e5400dea0197717b5b3d74b7`，按 MIT 许可证保留来源与声明。
- 将完整 Alpha158（158 个唯一特征、13 个家族）映射为 FactorFactory 原生 DSL。
- 新增 `ts_quantile`、`ts_slope`、`ts_rsquare`、`ts_resi`、`ts_argmax`、`ts_argmin`、逐元素比较与极值算子。
- 采用 Qlib 的 DataHandler / Dataset / Processor / Recorder 思路，但不绕过本系统的可执行标签、成本和冻结层。
- 使用训练安全的双向筛查，再对每个市场/组合模式的训练前十执行 HOLDOUT、Vault 和 2020 至最新评级。
- 保存 JSON、CSV、HTML、checkpoint 以及带 SHA256 的不可变 manifest。

## 在普通研究任务中的应用

“定义研究任务”现在保存任务级 `factorfactory.qlib-task-integration/v1` 配置。算法池类模板默认启用三条真实执行路径：

1. `qlib_alpha158_prior`：在结构搜索 30% 配额内部，按当前收益机制、字段兼容性和任务内未探索度，从固定版本 Alpha158 选择可审计种子。
2. `gbdt_residual_distill` 的 Qlib 候选池：在原生随机与在位者变异之外加入相容的 Alpha158 表达式，再由训练安全的 GBDT 元模型筛选并蒸馏为 DSL。
3. `qlib_joint_residual_distill`：建立逐日逐股票 Alpha158 可执行特征矩阵，Processor 只在 `INNER_PUBLIC` 拟合；LightGBM 使用严格扩展窗口 OOF，且可先用任务内最佳表达式解释标签、再学习剩余残差。稳定重要特征被蒸馏为最多 12 个短 DSL，随后从零进入 V4.2。

联合模型默认排除低保真 `VWAP0`，所以模型矩阵通常为 157 个 Alpha158 特征；目录与单因子审计仍完整保留 158 个。联合模型按研究子任务轮转刷新，并使用内容寻址 Parquet 缓存。`META_HOLDOUT`、`FACTOR_VAULT` 和冻结评级均不会进入模型、预算分配或蒸馏提示。

任务级动态治理还包括：实际/有效试验数、DSR/PBO、方向等价 AST 去重、信号秩 sketch 去重、同协议收益路径聚类，以及基于“唯一 Gate 改善 / CPU 秒”的自适应算法预算；每个研究组保留最低探索份额。

候选研究记录保存 `qlib_feature`、原始家族、上游提交、模型协议、蒸馏组件、候选数及 fidelity stage。Qlib 不改变 V4.2/V4.3 评分、方向冻结、成本、HOLDOUT 或 Vault。

历史任务没有该配置时保持原行为；结构化随机对照与直接 LLM 对照默认不启用 Qlib，避免破坏实验可比性。

## 没有照搬的部分

- 没有导入 Qlib 官方 benchmark 数字，也没有用 Qlib 默认标签替代本系统的 `t` 日信号、`t+1` 开盘成交与开盘到开盘持有期收益。
- 没有把官方 Alpha158 表达式的早期不完整窗口作为有效样本；本地要求完整窗口。
- 没有把 US `amount / vol` 当成可靠 VWAP。当前 US `amount` 主要是 `close × volume` 代理，`VWAP0` 因不可识别而 fail-closed。

## 可复现命令

```bash
POLARS_MAX_THREADS=5 .venv/bin/python backend/scripts/run_qlib_alpha158_benchmark.py \
  --markets ashare us \
  --us-modes long_only long_short \
  --universe-n 500 \
  --horizon 5 \
  --full-audit-top 10 \
  --run-id alpha158-dual-market-20260823
```

每个模式实际评价 158 个表达式，并在训练层同时评价正反两个方向，即每个模式 316 个方向假设；预声明检验次数保持 1000。

## 结果

| 市场/组合 | 完成 | 计算拒绝 | 训练通过 | 完整审计 | 完整通过 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A股纯多 | 158 | 0 | 0 | 10 | 0 |
| 美股纯多 | 158 | 1 | 0 | 10 | 0 |
| 美股多空 | 158 | 1 | 0 | 10 | 0 |

A股冻结评级前三：

| 因子 | 方向 | 训练 Gate | 学习分 | 评级年化 | 评级 Sharpe | 最大回撤 | 等级 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `BETA60` | -1 | 0.0453 | 1.1501 | 4.12% | 0.36 | 24.85% | F1 |
| `QTLU60` | +1 | 0.0295 | 0.9909 | 3.92% | 0.34 | 24.48% | F1 |
| `MA60` | +1 | 0.0055 | 0.9199 | 2.81% | 0.27 | 25.08% | F1 |

美股纯多冻结评级最高为 `STD60`：年化 6.42%、Sharpe 0.39、最大回撤 48.80%，但训练层方向调整后 ICIR 为负、HAC 与收益下置信界不合格，等级 F1。美股多空前十全部为负年化与负 Sharpe，换手和成本压力尤其严重。

## 科学结论

1. Qlib Alpha158 已成为可复用的候选语法库和基准协议，而不是一组默认有效的正式因子。
2. A股存在若干值得作为后续组合候选或残差搜索种子的长窗口结构，但当前没有任何一个满足正式入库门槛。
3. 本地美股数据与成本协议下，单因子 Alpha158 没有显示出可晋级证据；多空结果明显差于纯多。
4. `VWAP0` 曾因近常数截面和并列排序产生虚假高分。通用评价器现已在排序前拒绝近常数截面，US VWAP 还增加了来源级 fail-closed；原异常结果保留在质量拒绝记录中，未进入排名。
5. 后续若使用 Alpha158，应把它当作 LightGBM/CatBoost、残差 OOF 或 DSL 蒸馏的输入特征库，并继续使用相同冻结协议评价增量，而不能因“来自 Qlib”降低门槛。

## 工件

- 运行目录：`var/reports/qlib-alpha158/alpha158-dual-market-20260823/`
- 完整性清单：`manifest.json`
- 三个 HTML：`ashare-long_only-report.html`、`us-long_only-report.html`、`us-long_short-report.html`
