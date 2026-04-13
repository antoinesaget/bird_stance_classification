#!/usr/bin/env bash
# Purpose: Export annotations from the live Label Studio project on TrueNAS.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_cmd python3

ENV_FILE_REL="${ENV_FILE:-projects/labelstudio/deploy/env/truenas.env}"
ENV_FILE_PATH="$(resolve_repo_path "$ENV_FILE_REL")"
source_env "$ENV_FILE_PATH"

[[ -n "${PROJECT_ID:-}" ]] || die "PROJECT_ID is required"
[[ -n "${LABEL_STUDIO_API_TOKEN:-}" ]] || die "LABEL_STUDIO_API_TOKEN is required in $ENV_FILE_PATH"

EXPORT_STEM="${EXPORT_NAME:-${ANNOTATION_VERSION:-project_${PROJECT_ID}_$(date -u +%Y%m%dT%H%M%SZ)}}"
EXPORT_DIR="$BIRDS_DATA_ROOT/labelstudio/exports"
mkdir -p "$EXPORT_DIR"

CMD=(
  run_birdsys labelstudio export-snapshot
  --base-url "$LABEL_STUDIO_URL"
  --api-token "$LABEL_STUDIO_API_TOKEN"
  --project-id "$PROJECT_ID"
  --output "$EXPORT_DIR/${EXPORT_STEM}.json"
  --metadata-out "$EXPORT_DIR/${EXPORT_STEM}-info.json"
)

if [[ -n "${EXPORT_TITLE:-}" ]]; then
  CMD+=(--title "$EXPORT_TITLE")
fi
if [[ "${DOWNLOAD_ALL_TASKS:-1}" != "0" ]]; then
  CMD+=(--download-all-tasks)
fi

"${CMD[@]}"
