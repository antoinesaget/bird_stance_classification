#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

PYTHON_BIN="$(repo_python_bin)"

ENV_FILE_REL="${ENV_FILE:-deploy/env/iats.env}"
ENV_FILE_PATH="$(resolve_repo_path "$ENV_FILE_REL")"
source_env "$ENV_FILE_PATH"

BIRDS_DATA_ROOT="$(resolve_repo_path "$BIRDS_DATA_ROOT")"
[[ -n "${ANNOTATION_VERSION:-}" ]] || die "ANNOTATION_VERSION is required"

CMD=(
  "$PYTHON_BIN" scripts/make_crops.py
  --data-root "$BIRDS_DATA_ROOT"
  --annotation-version "$ANNOTATION_VERSION"
)

printf '[ops] %s\n' "${CMD[*]}"
"${CMD[@]}"

CMD=(
  "$PYTHON_BIN" scripts/build_dataset.py
  --data-root "$BIRDS_DATA_ROOT"
  --annotation-version "$ANNOTATION_VERSION"
  --train-pct "${TRAIN_PCT:-90}"
  --val-pct "${VAL_PCT:-0}"
  --test-pct "${TEST_PCT:-10}"
)
if [[ -n "${DATASET_VERSION:-}" ]]; then
  CMD+=(--dataset-version "$DATASET_VERSION")
fi

printf '[ops] %s\n' "${CMD[*]}"
"${CMD[@]}"
