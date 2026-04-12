#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

PYTHON_BIN="$(repo_python_bin)"

ENV_FILE_REL="${ENV_FILE:-deploy/env/iats.env}"
ENV_FILE_PATH="$(resolve_repo_path "$ENV_FILE_REL")"
source_env "$ENV_FILE_PATH"

BIRDS_DATA_ROOT="$(resolve_repo_path "$BIRDS_DATA_ROOT")"
export BIRDS_DATA_ROOT

resolve_path_hint() {
  local raw="$1"
  if [[ "$raw" = /* ]]; then
    printf '%s\n' "$raw"
    return
  fi
  if [[ -e "$REPO_ROOT/$raw" ]]; then
    printf '%s\n' "$REPO_ROOT/$raw"
    return
  fi
  printf '%s\n' "$BIRDS_DATA_ROOT/$raw"
}

if [[ -n "${DATASET_DIR:-}" ]]; then
  DATASET_DIR="$(resolve_path_hint "$DATASET_DIR")"
elif [[ -n "${DATASET_VERSION:-}" ]]; then
  DATASET_DIR="$BIRDS_DATA_ROOT/derived/datasets/${DATASET_VERSION}"
else
  die "DATASET_DIR or DATASET_VERSION is required"
fi

CHECKPOINT_PATH="${CHECKPOINT:-${MODEL_B_CHECKPOINT:-$BIRDS_DATA_ROOT/models/attributes/served/model_b/current}}"
CHECKPOINT_PATH="$(resolve_path_hint "$CHECKPOINT_PATH")"

CMD=(
  "$PYTHON_BIN" scripts/evaluate_model_b_checkpoint.py
  --data-root "$BIRDS_DATA_ROOT"
  --dataset-dir "$DATASET_DIR"
  --checkpoint "$CHECKPOINT_PATH"
  --split "${SPLIT:-train}"
)
if [[ -n "${EVAL_OUTPUT_DIR:-}" ]]; then
  CMD+=(--output-dir "$(resolve_path_hint "$EVAL_OUTPUT_DIR")")
fi
if [[ -n "${EVAL_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=( ${EVAL_ARGS} )
  CMD+=("${EXTRA_ARGS[@]}")
fi

printf '[ops] %s\n' "${CMD[*]}"
"${CMD[@]}"
