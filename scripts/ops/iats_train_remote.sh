#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_cmd python3
PYTHON_BIN="$(repo_python_bin)"

ENV_FILE_REL="${ENV_FILE:-deploy/env/iats.env}"
ENV_FILE_PATH="$(resolve_repo_path "$ENV_FILE_REL")"
source_env "$ENV_FILE_PATH"

BIRDS_DATA_ROOT="$(resolve_repo_path "$BIRDS_DATA_ROOT")"
export BIRDS_DATA_ROOT

if [[ -n "${TRAIN_CMD:-}" ]]; then
  log "Running custom training command on iats"
  bash -lc "$TRAIN_CMD"
  exit 0
fi

PIPELINE="${TRAIN_PIPELINE:-attributes}"
case "$PIPELINE" in
  attributes)
    if [[ -n "${DATASET_DIR:-}" ]]; then
      DATASET_DIR="$(resolve_repo_path "$DATASET_DIR")"
    elif [[ -n "${DATASET_VERSION:-}" ]]; then
      DATASET_DIR="$BIRDS_DATA_ROOT/derived/datasets/${DATASET_VERSION}"
    else
      die "DATASET_DIR or DATASET_VERSION is required for TRAIN_PIPELINE=attributes"
    fi

    CMD=("$PYTHON_BIN" scripts/train_attributes.py --data-root "$BIRDS_DATA_ROOT" --dataset-dir "$DATASET_DIR")
    if [[ "${TRAIN_SMOKE:-0}" == "1" ]]; then
      CMD+=(--smoke)
    fi
    if [[ -n "${TRAIN_ARGS:-}" ]]; then
      # shellcheck disable=SC2206
      EXTRA_ARGS=( ${TRAIN_ARGS} )
      CMD+=("${EXTRA_ARGS[@]}")
    fi
    ;;
  image-status)
    [[ -n "${ANNOTATION_VERSION:-}" ]] || die "ANNOTATION_VERSION is required for TRAIN_PIPELINE=image-status"
    CMD=("$PYTHON_BIN" scripts/train_image_status.py --data-root "$BIRDS_DATA_ROOT" --annotation-version "$ANNOTATION_VERSION")
    if [[ "${TRAIN_SMOKE:-0}" == "1" ]]; then
      CMD+=(--smoke)
    fi
    if [[ -n "${TRAIN_ARGS:-}" ]]; then
      # shellcheck disable=SC2206
      EXTRA_ARGS=( ${TRAIN_ARGS} )
      CMD+=("${EXTRA_ARGS[@]}")
    fi
    ;;
  *)
    die "Unsupported TRAIN_PIPELINE: $PIPELINE"
    ;;
esac

printf '[ops] %s\n' "${CMD[*]}"
"${CMD[@]}"
