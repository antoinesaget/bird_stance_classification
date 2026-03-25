#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_cmd python3

ENV_FILE_REL="${ENV_FILE:-deploy/env/truenas.env}"
ENV_FILE_PATH="$(resolve_repo_path "$ENV_FILE_REL")"
source_env "$ENV_FILE_PATH"

[[ -n "${LABEL_STUDIO_API_TOKEN:-}" ]] || die "LABEL_STUDIO_API_TOKEN is required"

SOURCE_PROJECT_ID="${SOURCE_PROJECT_ID:-${LABEL_STUDIO_SOURCE_PROJECT_ID:-4}}"
TARGET_PROJECT_TITLE="${TARGET_PROJECT_TITLE:-${LABEL_STUDIO_TARGET_PROJECT_TITLE:-Bird Schema V2}}"
LABEL_STUDIO_ML_BACKEND_URL="${LABEL_STUDIO_ML_BACKEND_URL:-http://192.168.0.42:9090}"
LABEL_STUDIO_ML_TITLE="${LABEL_STUDIO_ML_TITLE:-birdsys-schema-v2}"
REPORT_DIR="${BIRDS_DATA_ROOT}/labelstudio/projects"
mkdir -p "$REPORT_DIR"
REPORT_OUT="${REPORT_DIR}/$(printf '%s' "$TARGET_PROJECT_TITLE" | tr ' /' '__').json"

CMD=(
  python3 scripts/create_labelstudio_project.py
  --base-url "$LABEL_STUDIO_URL"
  --api-token "$LABEL_STUDIO_API_TOKEN"
  --label-config "$(resolve_repo_path "labelstudio/label_config.xml")"
  --source-project-id "$SOURCE_PROJECT_ID"
  --title "$TARGET_PROJECT_TITLE"
  --ml-url "$LABEL_STUDIO_ML_BACKEND_URL"
  --ml-title "$LABEL_STUDIO_ML_TITLE"
  --report-out "$REPORT_OUT"
)

printf '[ops] %s\n' "${CMD[*]}"
"${CMD[@]}"
