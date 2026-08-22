#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")"
export FF_PORT=10013
export FF_SERVICE_INSTANCE="factorfactory-full-llm-three-layer"
export FF_SERVICE_ARCHITECTURE="full_llm_three_layer"
export FF_AUTOSTART_RESEARCH=1
# The Linux 10013 runtime is restored into an isolated local database.  This
# prevents its PostgreSQL primary keys, engine state and audit history from
# colliding with the 10010/10011/10012 control-plane database.
export FF_DATABASE_URL="postgresql+asyncpg://jiangjingzhe@localhost:5432/factor_factory_10013"
export FF_BACKTEST_ARTIFACT_ROOT="$PWD/var/backtests-10013"
# Two workers may overlap LLM calls, while only one full-panel evaluation owns
# memory/CPU at a time. This coexists safely with the external leaderboard run.
export FF_MAX_PARALLEL_EVALUATIONS=1
export FF_LLM_TIMEOUT_SECONDS=420
export FF_LLM_MAX_ATTEMPTS=2
# New tasks have no historical nodes. Avoid blocking this independent service
# on migration of unrelated legacy feedback envelopes.
export FF_SKIP_HISTORICAL_FEEDBACK_BACKFILL=1
exec ./service.sh "${1:-status}" "${2:-}"
