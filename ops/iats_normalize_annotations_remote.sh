#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

PYTHON_BIN="$(repo_python_bin)"

ENV_FILE_REL="${ENV_FILE:-projects/ml_backend/deploy/env/iats.env}"
ENV_FILE_PATH="$(resolve_repo_path "$ENV_FILE_REL")"
source_env "$ENV_FILE_PATH"

BIRDS_DATA_ROOT="$(resolve_repo_path "$BIRDS_DATA_ROOT")"
[[ -n "${ANNOTATION_VERSION:-}" ]] || die "ANNOTATION_VERSION is required"

if [[ -n "${EXPORT_JSON:-}" ]]; then
  if [[ "$EXPORT_JSON" = /* ]]; then
    EXPORT_JSON_PATH="$EXPORT_JSON"
  else
    EXPORT_JSON_PATH="$BIRDS_DATA_ROOT/labelstudio/exports/$EXPORT_JSON"
  fi
elif [[ -n "${EXPORT_NAME:-}" ]]; then
  EXPORT_JSON_PATH="$BIRDS_DATA_ROOT/labelstudio/exports/${EXPORT_NAME}.json"
else
  EXPORT_JSON_PATH="$BIRDS_DATA_ROOT/labelstudio/exports/${ANNOTATION_VERSION}.json"
fi

[[ -f "$EXPORT_JSON_PATH" ]] || die "Missing export json: $EXPORT_JSON_PATH"

CMD=(
  run_birdsys datasets normalize-export
  --data-root "$BIRDS_DATA_ROOT"
  --export-json "$EXPORT_JSON_PATH"
  --annotation-version "$ANNOTATION_VERSION"
)

printf '[ops] %s\n' "${CMD[*]}"
"${CMD[@]}"
