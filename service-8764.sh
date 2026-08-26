#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")"

logical_cores="$(sysctl -n hw.logicalcpu 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || print 2)"
half_cores=$(( logical_cores / 2 ))
(( half_cores < 1 )) && half_cores=1

export FF_PORT="${FF_PORT:-8764}"
export FF_HOST="${FF_HOST:-127.0.0.1}"
export FF_SERVICE_INSTANCE="${FF_SERVICE_INSTANCE:-factorfactory-8764-rust-mirror}"
export FF_SERVICE_ARCHITECTURE="${FF_SERVICE_ARCHITECTURE:-rust-shadow-mirror-v1}"
export FF_AUTOSTART_RESEARCH="${FF_AUTOSTART_RESEARCH:-0}"
export FF_BACKTEST_BACKEND="${FF_BACKTEST_BACKEND:-rust_shadow}"
export FF_BACKTEST_ARTIFACT_ROOT="${FF_BACKTEST_ARTIFACT_ROOT:-$PWD/var/backtests-8764-rust}"
export FF_MAX_PARALLEL_EVALUATIONS="${FF_MAX_PARALLEL_EVALUATIONS:-1}"
export POLARS_MAX_THREADS="${POLARS_MAX_THREADS:-$half_cores}"

if [[ ! -f "$PWD/rust-backtest-kernel/target/release/libfactorfactory_rust_backtest.dylib" \
      && ! -f "$PWD/rust-backtest-kernel/target/release/libfactorfactory_rust_backtest.so" ]]; then
  "$PWD/scripts/build_rust_backtest_kernel.sh"
fi

exec "$PWD/service.sh" "$@"
