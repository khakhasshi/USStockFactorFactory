# 双层与三层理想连续研究服务 V1

## 服务边界

| 端口 | 实例 | 架构 | 常驻任务 |
|---|---|---|---|
| 10011 | `factorfactory-two-layer-ideal` | 算法池 → Researcher LLM | 美股 45、A股 46 |
| 10012 | `factorfactory-three-layer-ideal` | 算法池 → Researcher LLM → Governor LLM | 美股 47、A股 48 |

服务只监听 `127.0.0.1`。任务通过 `service_instance` 与端口绑定；错误架构或错误实例的任务会在 worker 启动前被拒绝。各端口使用独立的当前任务设置，任务列表只展示属于该服务的两个任务。

## 第一层：异质算法池

第一层同时保留四类互补搜索器：

- `structured_random`：受 DSL、机制族与字段白名单约束的广泛探索；
- `evolutionary`：围绕训练安全的高质量节点做结构变异；
- `surrogate_kernel`：利用历史表达式与得分的局部代理排序候选；
- `q_learning`：按机制族和失败反馈更新算子选择。

算法池负责覆盖率与候选多样性，不接触评级窗口。每个候选保留算法、种子、父节点、机制、深度和上下文指纹。

## 第二层：Researcher LLM

Researcher 接收第一层种子、当前任务约束和同协议训练安全反馈，批量完成机制审查和一次可归因改进。它不能改变市场、持仓模式、成本、评估器、数据分层、候选预算或评级规则。输出必须通过 DSL、字段、机制一致性、相似度和语义检查；不合格响应记为 `rejected`，不会伪装成成功的 LLM 改进。

`memory_mode=adaptive` 使用任务内连续记忆。记忆只来自 V4.2 训练安全评价、失败原因、重复率和已冻结方向，不包含 Frozen Rating V4.3。

## 第三层：Governor LLM

10012 的 Governor 只优化 Researcher 的声明式搜索策略。前 30 个候选构成确定性冷启动 cohort；其后 Governor 每个外层周期最多修改一个字段，并预先声明：

- 诊断和证据；
- 可证伪假设与预期指标；
- 证伪规则、置信度和有效 cohort 数；
- 单变量修改字段。

修改后采用相同任务、种子、候选数和成本的配对 cohort 比较；接受规则使用预注册的单侧配对检验，并要求硬门槛通过率不退化。网络失败不会用随机 Governor 决策冒充。

## 无限预算与 7×24

四项任务统一设置：

- `continuous_operation=true`；
- `candidate_evaluation_budget=0`；
- `target_factor_count=0`；
- `budget_policy.mode=unlimited`；
- `service_autostart=true`。

这里的 `0` 明确表示无停止上限，不表示不执行。外层步数、运行小时和 LLM 调用数保留为 cohort 诊断尺度，但在连续模式下不触发停止。服务重启或 worker 故障后，监督器以 30–900 秒指数退避恢复；已提交节点、试验、审计和研究记录不会删除。

每个服务进程最多同时执行一个重型全面板评价，两个市场任务可以并行准备上下文和等待 LLM。LLM 请求每 15 秒更新一次运行心跳，默认超时 420 秒，最多进行两次可审计的传输级重试。

## 评级与数据边界

研究评价使用 V4.2，先在训练安全层同时检查 `+1/-1`，计入双向搜索惩罚并冻结方向。正式 Frozen Rating V4.3 独立计算：

- 起点固定为 `2020-01-01`；
- 终点为对应面板 2026 年最新可用交易日；
- 美股当前为 `2026-08-19`，A股当前为 `2026-08-20`；
- 评级覆盖 2020 至最新数据，不受旧 `FACTOR_VAULT` 标签终点限制；
- 评级结果不回流 Researcher 或 Governor。

按当前全局研究政策，PIT 质量只作披露，不作为快速研究或因子入库门槛。

## 运维命令

```bash
./service-10011.sh status
./service-10012.sh status
./service-10011.sh logs
./service-10012.sh logs
```

重新生成任务清单默认只预览；`--apply` 幂等写入。脚本拒绝覆盖已有研究产物的冲突任务：

```bash
.venv/bin/python backend/scripts/create_continuous_ideal_services.py
.venv/bin/python backend/scripts/create_continuous_ideal_services.py --apply
```
