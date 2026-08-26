#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")"

logical_cores="$(sysctl -n hw.logicalcpu 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || print 2)"
half_cores=$(( logical_cores / 2 ))
(( half_cores < 1 )) && half_cores=1

export FF_PORT="${FF_PORT:-8765}"
export FF_HOST="${FF_HOST:-127.0.0.1}"
export FF_SERVICE_INSTANCE="${FF_SERVICE_INSTANCE:-factorfactory-8765}"
export FF_AUTOSTART_RESEARCH="${FF_AUTOSTART_RESEARCH:-0}"
# Full panel evaluations are memory intensive. Two concurrent evaluations share
# one five-thread Polars pool; this caps CPU without multiplying panel memory by 5.
export FF_MAX_PARALLEL_EVALUATIONS="${FF_MAX_PARALLEL_EVALUATIONS:-2}"
export POLARS_MAX_THREADS="${POLARS_MAX_THREADS:-$half_cores}"

exec "$PWD/service.sh" "$@"
