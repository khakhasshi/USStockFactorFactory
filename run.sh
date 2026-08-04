#!/bin/zsh
# USStockFactorFactory 启动脚本
# 端口默认为 9999 (可用 FF_PORT 覆盖)
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -r backend/requirements.txt
cd backend
exec ../.venv/bin/python -m app.main
