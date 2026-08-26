#!/bin/zsh
set -euo pipefail

repo_root="${0:A:h:h}"
cd "$repo_root/rust-backtest-kernel"

cargo test
cargo build --release

"$repo_root/.venv/bin/python" - <<'PY'
from backend.app.backtest.rust_kernel import reload_rust_kernel, rust_kernel_capabilities

reload_rust_kernel()
status = rust_kernel_capabilities()
if not status["available"]:
    raise SystemExit(status["load_error"])
print(status)
PY
