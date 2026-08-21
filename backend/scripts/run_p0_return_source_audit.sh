#!/bin/zsh
set -euo pipefail

project_root="${0:A:h:h:h}"
output_dir="$project_root/var/audits/p0-return-source-diversity-v1"

cd "$project_root"
exec /usr/bin/env \
  PYTHONPATH="$project_root:$project_root/backend" \
  POLARS_MAX_THREADS=2 \
  python3 -u backend/scripts/factor_return_source_audit.py \
  --source-report var/reports/ashare-long-only-vector-screen-2020-latest-20260807-133714 \
  --source-report var/reports/us-long-short-vector-screen-2020-latest-20260807-133714 \
  --source-report var/reports/us-long-only-vector-screen-2020-latest-20260807-133714 \
  --output-dir "$output_dir" \
  --workers 3 \
  --threads-per-worker 2 \
  --correlation-threshold 0.80

