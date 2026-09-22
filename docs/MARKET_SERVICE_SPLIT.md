# 本机 A 股 / 美股研究服务拆分

2026-09-16 按市场拆分；研究计算、评分、回测协议不变，实盘服务和云端均不在本次变更范围。

| 项目 | 美股 | A 股 |
|---|---|---|
| 项目目录 | `/Users/jiangjingzhe/Finance_Data_Center/releases/source_roots/Portfolios_USStockFactorFactory` | `/Users/jiangjingzhe/Portfolios/AShareFactorFactory` |
| 本机端口 | `127.0.0.1:8765` | `127.0.0.1:8767` |
| PostgreSQL 数据库 | `factor_factory_us_8765` | `factor_factory_ashare_8767` |
| 选中的历史任务 | #58 美股QLIB | #57 A股QLIB |
| 事件回测文件 | `var/backtests-us` | `var/backtests` |

两套目录均使用 `./service.sh start|stop|restart|status|logs`。`run.sh` 和
`service.sh` 均读取本项目 `.service.env`，因此直接启动不会误连原混合数据库。
只监听回环地址；启动服务不会自动启动历史挖掘任务。进入页面选择任务后可手动恢复。

数据库按 `experiments.research_config.market` 拆分，缺失 market 的早期 #1/#2
按原有系统默认美股处理。任务和各类研究记录保留原 ID、创建时间、表达式和评分；
全局无任务归属的事件/LLM审计及研究设置复制到两库。旧交易模块数据不迁入研究分库。
LLM 接入配置随研究设置迁移，不在报告中输出凭据。

复制每张表时使用同一只读重复读快照，并对有序 PostgreSQL 二进制 COPY
流计算 SHA-256，目标回读必须一致。唯一有意改变的设置是默认选中任务。
原 `factor_factory` 数据库完整保留且不再由这两套服务使用；未删除任何原研究记录。
旧的 `factor_factory_ashare`、`factor_factory_10013` 等数据库不在本次迁移范围。

原回测产物按市场 ID 列表复制并逐文件校验；旧向量回测无事件文件会单独记入清单。
原始 manifest 和路径溯源不重写。不可变快照、历史报告保留副本；榜单入口按服务市场过滤。
行情面板仍读取 Finance_Data_Center 既有数据源，不复制、不改写原行情。
独立环境复用已安装依赖的 APFS 写时复制副本，未共享可写虚拟环境。

迁移备份及校验清单位于美股项目 `var/migrations/market-split-20260916/`：

- `factor_factory.pre-split.dump`：完整迁移前备份（目录700、文件600）
- `verified-copy.json`：逐表行数/内容校验、ID清单
- `verified-artifacts.json`：回测文件复制结果、源端缺失文件记录

回滚：先停止两套服务，确认无计算任务；将美股 `.service.env` 中数据库改回
`factor_factory`、回测目录改回原 `var/backtests`，清空 `FF_SERVICE_MARKET`
以恢复混合市场服务，再启动美股目录服务。两分库保留，不执行 DROP/DELETE。
回滚前应先归档拆分后新产生的记录，不能把旧库当成包含最新工作的数据库。
