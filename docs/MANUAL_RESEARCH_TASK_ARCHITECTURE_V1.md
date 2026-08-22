# 手工研究任务架构 V1

手工任务不再通过一个含义模糊的 `proposal_mode` 猜测实际运行方式。创建时必须选择架构模板或明确的自定义层级；服务端再次解析并冻结以下字段：

- `architecture_schema` 与 `architecture_template`
- `layer1_enabled`、`layer2_enabled`、`layer3_enabled`
- `search_algorithms`、`memory_mode` 与服务端推导的 `proposal_mode`
- 任务级 `engine_config`、评价配置、候选预算和训练收益来源治理

内置模板：

| 模板 | 第一层 | 第二层 | 第三层 | 适用目的 |
|---|---|---|---|---|
| `random_only` | 结构化随机 | 关闭 | 关闭 | 无 LLM 随机基线 |
| `algorithm_pool_only` | 随机、进化、代理模型、Q-learning | 关闭 | 关闭 | 第一层算法组合 |
| `random_researcher` | 结构化随机 | Researcher LLM | 关闭 | 用户要求的“随机 → LLM”主模板 |
| `algorithm_pool_researcher` | 完整算法池 | Researcher LLM | 关闭 | 两层搜索主架构 |
| `full_three_layer` | 完整算法池 | Researcher LLM | Governor LLM | 三层完整架构 |
| `direct_researcher` | 关闭 | Researcher LLM | 关闭 | 直接 LLM 对照 |

约束：第三层必须依赖第二层；启用第一层时至少选择一种合法算法；无 LLM 模板强制冷记忆；LLM provider 未配置时任务允许先保存，但 worker 启动即熔断，不会无声退化成随机结果。历史 API 任务保留 `legacy` 解释，避免修改已有研究语义。

入口为控制台“设置 → 定义研究任务”。实验页只负责查看、启动、停止、归档和改名，避免继续产生缺少评价、架构和预算声明的“基础任务”。
