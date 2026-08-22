# 20010 内部研究增强 Demo

该分支只提高内部因子研究能力，不包含产品化、社区、比赛、商店或收费能力。

## 架构决定

- 连续发现循环：0 层强制 LLM。结构化随机、进化、代理模型、Q-learning 和残差路径 Beam 在 LLM 供应商不可用时仍可运行。
- 短名单审查：2 个按需、相互隔离的 LLM reviewer。方法审查与代码审查通过 packet hash 防止看到对方意见；LLM 不可用时标记 pending，不使研究 worker 崩溃。
- 调度治理：确定性程序管理实际试验数、动态多重检验、DSR、PBO/CSCV、Harvey-Liu haircut、Winner's Curse 和机制覆盖，不把调度器伪装成第三层 LLM。

## 研究安全边界

- Residual OOF Beam 的精确入口要求调用者提供时间对齐的 OOF incumbent/candidate prediction matrix。
- 7×24 搜索中只能使用 PUBLIC + META_TRAIN 的压缩收益路径作为低相似路径调度代理；正式晋级必须补做精确 OOF。
- 冻结评级从 2020-01-01 到最新可用日，结果不得回流搜索或 LLM。
- 119 个 Lens 被清洗为可证伪研究问题，33 个基础机制只作搜索语法补缺；两者都不是成品因子，也不生成基础机制、变换和窗口的笛卡尔积。
- 文档输入只提供可追溯创意，DSL 候选不获得分数加成，必须进入同一评价器。

## API

- `GET /api/research-intelligence/mechanisms`
- `POST /api/research-intelligence/document-to-dsl`
- `POST /api/research-intelligence/residual-beam`
- `POST /api/research-intelligence/overfit-diagnostics`
- `GET /api/research-intelligence/trial-ledger?experiment_id=...`
- `POST /api/research-intelligence/blind-review/packets`
- `POST /api/research-intelligence/blind-review/seal`
