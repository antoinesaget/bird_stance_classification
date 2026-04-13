#!/usr/bin/env bash
# Purpose: Import a prepared lines-project task bundle into Label Studio on TrueNAS.
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

TASKS_JSON="$(resolve_repo_path "${LINES_TASKS_JSON:-${LINES_DATA_ROOT:-/mnt/tank/media/lines_project}/${LINES_IMPORT_RELATIVE_ROOT}/${LINES_BATCH_NAME}.tasks.json}")"
REPORT_OUT="$(resolve_repo_path "${LINES_IMPORT_REPORT_OUT:-${LINES_DATA_ROOT:-/mnt/tank/media/lines_project}/${LINES_IMPORT_RELATIVE_ROOT}/${LINES_BATCH_NAME}.import-report.json}")"

log "Importing task bundle '$LINES_BATCH_NAME' into Label Studio project '$LINES_PROJECT_ID'"
run_birdsys labelstudio import-tasks \
  --base-url "$LABEL_STUDIO_URL" \
  --api-token "$LABEL_STUDIO_API_TOKEN" \
  --project-id "$LINES_PROJECT_ID" \
  --tasks-json "$TASKS_JSON" \
  --report-out "$REPORT_OUT"
