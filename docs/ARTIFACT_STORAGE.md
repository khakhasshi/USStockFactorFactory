# 回测产物存储

新回测只持久化 Parquet 账本；交割单、往返交易与因子归因的 CSV 经原下载接口按需流式生成，
不创建常驻 CSV 或临时磁盘缓存。清单仍保留 CSV 文件名、行数和 SHA-256，同时标记
`storage=on_demand`、Parquet 来源与序列化器版本。Polars 版本已锁定在 backend/requirements.txt。

消费代码不应假设 manifest 中的所有文件都已物化；CSV 使用下载接口或
`backend.app.artifact_exports.csv_chunks(path)`。直接读取账本应使用 Parquet。

历史清单保持原样。已有 CSV 原路径继续可读；2026-09-11 的空间优化使用 macOS 透明压缩，
压缩前后字节哈希相同，静态研究报告链接也保持有效。可选的 `.csv.gz` 兼容读取可恢复原始字节。
复制到不支持 macOS 透明压缩的文件系统时，文件会以正常内容保存，空间可能重新增大。

重复文件使用 APFS clone 共享数据块并保持独立 inode；写入时分离，不把原始写入路径变成共享硬链接。
