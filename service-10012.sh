#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")"
export FF_PORT=10012
export FF_SERVICE_INSTANCE="factorfactory-three-layer-ideal"
export FF_SERVICE_ARCHITECTURE="three_layer"
export FF_AUTOSTART_RESEARCH=1
export FF_MAX_PARALLEL_EVALUATIONS=1
export FF_LLM_TIMEOUT_SECONDS=420
export FF_LLM_MAX_ATTEMPTS=2
exec ./service.sh "${1:-status}" "${2:-}"
