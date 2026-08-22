#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")"
export FF_PORT=20010
export FF_SERVICE_INSTANCE="factorfactory-internal-research-demo"
export FF_SERVICE_ARCHITECTURE="internal_research_hybrid"
export FF_AUTOSTART_RESEARCH=1
export FF_DATABASE_URL="postgresql+asyncpg://jiangjingzhe@localhost:5432/factor_factory_20010"
export FF_BACKTEST_ARTIFACT_ROOT="$PWD/var/backtests-20010"
export FF_PYTHON="/Users/jiangjingzhe/Finance_Data_Center/releases/source_roots/Portfolios_USStockFactorFactory/.venv/bin/python"
# Two research workers share one full-panel evaluation slot. Candidate
# generation and statistics may overlap, while the memory-heavy scan cannot.
export FF_MAX_PARALLEL_EVALUATIONS=1
export FF_LLM_TIMEOUT_SECONDS=420
export FF_LLM_MAX_ATTEMPTS=1
export FF_SKIP_HISTORICAL_FEEDBACK_BACKFILL=1
exec ./service.sh "${1:-status}" "${2:-}"
