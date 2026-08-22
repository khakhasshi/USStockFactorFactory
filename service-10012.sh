#!/bin/zsh
set -euo pipefail

print -u2 "10012 已退役：三层理想任务及完整研究历史已迁移到 10010。"
print -u2 "请使用 ./service.sh ${1:-status} 管理统一服务。"
exit 3
