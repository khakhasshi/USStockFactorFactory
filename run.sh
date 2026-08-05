#!/bin/zsh
# FactorFactory 启动脚本
# 默认仅监听 127.0.0.1:10010，可用 FF_HOST / FF_PORT 覆盖。
set -euo pipefail

cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv

requirements_file="backend/requirements.txt"
requirements_stamp=".venv/.factorfactory-requirements.sha256"
requirements_hash="$(shasum -a 256 "$requirements_file" | awk '{print $1}')"
installed_hash=""
[ ! -f "$requirements_stamp" ] || installed_hash="$(<"$requirements_stamp")"

if [[ "$installed_hash" != "$requirements_hash" ]] || ! .venv/bin/python -c \
  "import asyncpg, fastapi, httpx, numpy, polars, psutil, sqlalchemy, uvicorn" \
  >/dev/null 2>&1; then
  .venv/bin/python -m pip install -q -r "$requirements_file"
  print -r -- "$requirements_hash" > "$requirements_stamp"
fi

cd backend
exec ../.venv/bin/python -m app.main
