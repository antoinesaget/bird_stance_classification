#!/usr/bin/env bash
# Purpose: Build the compressed image mirror and task bundle for a lines-project import on TrueNAS.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

PYTHON_BIN="$(repo_python_bin)"

ENV_FILE_REL="${ENV_FILE:-projects/labelstudio/deploy/env/truenas.env}"
ENV_FILE_PATH="$(resolve_repo_path "$ENV_FILE_REL")"
source_env "$ENV_FILE_PATH"

LINES_DATA_ROOT="${LINES_DATA_ROOT:-/mnt/tank/media/lines_project}"
LINES_SOURCE_RELATIVE_ROOT="${LINES_SOURCE_RELATIVE_ROOT:-raw_images}"
LINES_IMPORT_RELATIVE_ROOT="${LINES_IMPORT_RELATIVE_ROOT:-labelstudio/imports}"
LINES_SAMPLE_SIZE="${LINES_SAMPLE_SIZE:-5000}"
LINES_SAMPLE_MODE="${LINES_SAMPLE_MODE:-random}"
LINES_SAMPLE_SEED="${LINES_SAMPLE_SEED:-20260325}"
LINES_JPEG_QUALITY="${LINES_JPEG_QUALITY:-60}"
LINES_MIRROR_RELATIVE_ROOT="${LINES_MIRROR_RELATIVE_ROOT:-labelstudio/images_compressed/lines_bw_stilts_q${LINES_JPEG_QUALITY}}"
LINES_DATASET_NAME="${LINES_DATASET_NAME:-lines_project}"
LINES_BATCH_NAME="${LINES_BATCH_NAME:-lines_bw_stilts_${LINES_SAMPLE_SIZE}_seed_${LINES_SAMPLE_SEED}_q${LINES_JPEG_QUALITY}}"
LINES_RECURSIVE="${LINES_RECURSIVE:-0}"

CMD=(
  run_birdsys labelstudio build-localfiles-batch
  --data-root "$LINES_DATA_ROOT"
  --source-relative-root "$LINES_SOURCE_RELATIVE_ROOT"
  --mirror-relative-root "$LINES_MIRROR_RELATIVE_ROOT"
  --import-relative-root "$LINES_IMPORT_RELATIVE_ROOT"
  --batch-name "$LINES_BATCH_NAME"
  --sample-size "$LINES_SAMPLE_SIZE"
  --sample-mode "$LINES_SAMPLE_MODE"
  --seed "$LINES_SAMPLE_SEED"
  --jpeg-quality "$LINES_JPEG_QUALITY"
  --dataset-name "$LINES_DATASET_NAME"
)

if [[ "$LINES_RECURSIVE" == "1" ]]; then
  CMD+=(--recursive)
fi

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  CMD+=(--overwrite)
fi

log "Preparing lines_project Label Studio batch '$LINES_BATCH_NAME'"
"${CMD[@]}"
