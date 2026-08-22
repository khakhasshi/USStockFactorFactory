#!/usr/bin/env bash
set -euo pipefail

project_root="${HOME}/Finance_Data_Center/releases/source_roots/Portfolios_USStockFactorFactory"
deploy_root="${project_root}/deploy/10013-linux"

mkdir -p "${project_root}/var/log" "${project_root}/var/backtests"
umask 077
if [[ ! -s "${deploy_root}/.postgres.env" ]]; then
  postgres_password="$(openssl rand -hex 24)"
  printf '%s\n' \
    'POSTGRES_USER=factorfactory' \
    'POSTGRES_DB=factor_factory' \
    "POSTGRES_PASSWORD=${postgres_password}" \
    > "${deploy_root}/.postgres.env"
fi

postgres_password="$(sed -n 's/^POSTGRES_PASSWORD=//p' "${deploy_root}/.postgres.env")"
printf '%s\n' \
  'FF_HOST=0.0.0.0' \
  'FF_PORT=10013' \
  'FF_SERVICE_INSTANCE=factorfactory-full-llm-three-layer' \
  'FF_SERVICE_ARCHITECTURE=full_llm_three_layer' \
  'FF_AUTOSTART_RESEARCH=1' \
  'FF_ALLOW_REMOTE_UNAUTHENTICATED=1' \
  'FF_MAX_PARALLEL_EVALUATIONS=1' \
  'FF_MAX_PARALLEL_PANEL_LOADS=1' \
  'FF_LLM_TIMEOUT_SECONDS=420' \
  'FF_LLM_MAX_ATTEMPTS=2' \
  'FF_SKIP_HISTORICAL_FEEDBACK_BACKFILL=1' \
  'FF_PANEL_AUTO_RELOAD=0' \
  'FF_DISABLE_FDC_DEFAULTS=1' \
  "FF_DATABASE_URL=postgresql+asyncpg://factorfactory:${postgres_password}@postgres:5432/factor_factory" \
  'FF_BACKTEST_ARTIFACT_ROOT=/app/var/backtests' \
  'POLARS_MAX_THREADS=4' \
  'MALLOC_ARENA_MAX=2' \
  > "${deploy_root}/.runtime.env"
chmod 600 "${deploy_root}/.postgres.env" "${deploy_root}/.runtime.env"

docker compose -f "${deploy_root}/compose.yml" up -d postgres
docker compose -f "${deploy_root}/compose.yml" build app
