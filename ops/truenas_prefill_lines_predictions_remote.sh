#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

PYTHON_BIN="$(repo_python_bin)"

ENV_FILE_REL="${ENV_FILE:-projects/labelstudio/deploy/env/truenas.env}"
ENV_FILE_PATH="$(resolve_repo_path "$ENV_FILE_REL")"
source_env "$ENV_FILE_PATH"

[[ -n "${LABEL_STUDIO_API_TOKEN:-}" ]] || die "LABEL_STUDIO_API_TOKEN is required"
[[ -n "${LABEL_STUDIO_URL:-}" ]] || die "LABEL_STUDIO_URL is required"

LINES_IMPORT_RELATIVE_ROOT="${LINES_IMPORT_RELATIVE_ROOT:-labelstudio/imports}"
LINES_SAMPLE_SIZE="${LINES_SAMPLE_SIZE:-5000}"
LINES_SAMPLE_SEED="${LINES_SAMPLE_SEED:-20260325}"
LINES_JPEG_QUALITY="${LINES_JPEG_QUALITY:-60}"
LINES_BATCH_NAME="${LINES_BATCH_NAME:-lines_bw_stilts_${LINES_SAMPLE_SIZE}_seed_${LINES_SAMPLE_SEED}_q${LINES_JPEG_QUALITY}}"
LINES_PROJECT_ID="${LINES_PROJECT_ID:-7}"
LINES_ML_BACKEND_URL="${LINES_ML_BACKEND_URL:-http://192.168.0.42:9090}"
LINES_TASK_PAGE_SIZE="${LINES_TASK_PAGE_SIZE:-100}"
LINES_PREDICT_BATCH_SIZE="${LINES_PREDICT_BATCH_SIZE:-16}"
LINES_PREDICTION_IMPORT_BATCH_SIZE="${LINES_PREDICTION_IMPORT_BATCH_SIZE:-100}"
LINES_LIMIT="${LINES_LIMIT:-0}"
LINES_ONLY_MISSING="${LINES_ONLY_MISSING:-1}"

REPORT_OUT="$(resolve_repo_path "${LINES_PREDICTION_REPORT_OUT:-${LINES_DATA_ROOT:-/mnt/tank/media/lines_project}/${LINES_IMPORT_RELATIVE_ROOT}/${LINES_BATCH_NAME}.predictions-report.json}")"

log "Pre-generating predictions for project '$LINES_PROJECT_ID' from '$LINES_ML_BACKEND_URL'"

ARGS=(
  --base-url "$LABEL_STUDIO_URL"
  --api-token "$LABEL_STUDIO_API_TOKEN"
  --project-id "$LINES_PROJECT_ID"
  --ml-backend-url "$LINES_ML_BACKEND_URL"
  --task-page-size "$LINES_TASK_PAGE_SIZE"
  --predict-batch-size "$LINES_PREDICT_BATCH_SIZE"
  --import-batch-size "$LINES_PREDICTION_IMPORT_BATCH_SIZE"
  --limit "$LINES_LIMIT"
  --report-out "$REPORT_OUT"
)

if [[ "$LINES_ONLY_MISSING" != "0" ]]; then
  ARGS+=(--only-missing)
fi

run_birdsys labelstudio prefill-predictions "${ARGS[@]}"
