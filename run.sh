#!/bin/zsh
# FactorFactory 启动脚本
# 默认仅监听 127.0.0.1:10010，可用 FF_HOST / FF_PORT 覆盖。
set -euo pipefail

cd "$(dirname "$0")"
runtime_python="${FF_PYTHON:-$PWD/.venv/bin/python}"
if [[ -z "${FF_PYTHON:-}" ]]; then
  [ -d .venv ] || python3 -m venv .venv
fi

requirements_file="backend/requirements.txt"
requirements_stamp=".venv/.factorfactory-requirements.sha256"
requirements_hash="$(shasum -a 256 "$requirements_file" | awk '{print $1}')"
installed_hash=""
[ ! -f "$requirements_stamp" ] || installed_hash="$(<"$requirements_stamp")"

if [[ -z "${FF_PYTHON:-}" ]] && { [[ "$installed_hash" != "$requirements_hash" ]] || ! "$runtime_python" -c \
  "import asyncpg, fastapi, httpx, numpy, polars, psutil, sqlalchemy, uvicorn" \
>/dev/null 2>&1; }; then
  "$runtime_python" -m pip install -q -r "$requirements_file"
  print -r -- "$requirements_hash" > "$requirements_stamp"
fi

cd backend
exec "$runtime_python" -m app.main
