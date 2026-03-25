#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_cmd rsync ssh

TRUENAS_HOST="${TRUENAS_HOST:-truenas}"
TRUENAS_REPO_ROOT="${TRUENAS_REPO_ROOT:-/mnt/apps/code/bird_stance_classification}"
IATS_HOST="${IATS_HOST:-iats}"
IATS_REPO_ROOT="${IATS_REPO_ROOT:-/home/antoine/bird_stance_classification}"
TRUENAS_BIRDS_DATA_ROOT="${TRUENAS_BIRDS_DATA_ROOT:-/mnt/tank/media/birds_project}"
IATS_BIRDS_DATA_ROOT="${IATS_BIRDS_DATA_ROOT:-${IATS_REPO_ROOT}/data/birds_project}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

if [[ -n "${PROJECT_ID:-}" ]]; then
  REMOTE_REPO_URL="${REMOTE_REPO_URL:-$(git -C "$REPO_ROOT" config --get remote.origin.url)}" \
  DEPLOY_BRANCH="${DEPLOY_BRANCH:-main}" \
  PROJECT_ID="${PROJECT_ID}" \
  ANNOTATION_VERSION="${ANNOTATION_VERSION:-}" \
  EXPORT_NAME="${EXPORT_NAME:-}" \
  EXPORT_TITLE="${EXPORT_TITLE:-}" \
  DOWNLOAD_ALL_TASKS="${DOWNLOAD_ALL_TASKS:-1}" \
  "$REPO_ROOT/scripts/ops/remote_repo_exec.sh" "$TRUENAS_HOST" "$TRUENAS_REPO_ROOT" scripts/ops/truenas_export_annotations_remote.sh
fi

ssh -o BatchMode=yes -o ConnectTimeout=15 "$IATS_HOST" "mkdir -p '$IATS_BIRDS_DATA_ROOT/labelstudio/exports' '$IATS_BIRDS_DATA_ROOT/metadata'"

EXPORT_STEM="${EXPORT_NAME:-${ANNOTATION_VERSION:-}}"
if [[ -n "$EXPORT_STEM" ]]; then
  log "Syncing export ${EXPORT_STEM} to iats"
  rsync -az "${TRUENAS_HOST}:${TRUENAS_BIRDS_DATA_ROOT}/labelstudio/exports/${EXPORT_STEM}.json" \
    "${TMP_DIR}/"
  rsync -az "${TMP_DIR}/${EXPORT_STEM}.json" \
    "${IATS_HOST}:${IATS_BIRDS_DATA_ROOT}/labelstudio/exports/"
  rsync -az "${TRUENAS_HOST}:${TRUENAS_BIRDS_DATA_ROOT}/labelstudio/exports/${EXPORT_STEM}-info.json" \
    "${TMP_DIR}/" || true
  [[ -f "${TMP_DIR}/${EXPORT_STEM}-info.json" ]] && \
    rsync -az "${TMP_DIR}/${EXPORT_STEM}-info.json" "${IATS_HOST}:${IATS_BIRDS_DATA_ROOT}/labelstudio/exports/" || true
else
  log "Syncing full export directory to iats"
  rsync -az "${TRUENAS_HOST}:${TRUENAS_BIRDS_DATA_ROOT}/labelstudio/exports/" "${TMP_DIR}/exports/"
  rsync -az "${TMP_DIR}/exports/" "${IATS_HOST}:${IATS_BIRDS_DATA_ROOT}/labelstudio/exports/"
fi

rsync -az "${TRUENAS_HOST}:${TRUENAS_BIRDS_DATA_ROOT}/metadata/images.parquet" "${TMP_DIR}/" || true
[[ -f "${TMP_DIR}/images.parquet" ]] && \
  rsync -az "${TMP_DIR}/images.parquet" "${IATS_HOST}:${IATS_BIRDS_DATA_ROOT}/metadata/" || true

if [[ "${NORMALIZE_ON_IATS:-1}" != "0" && -n "${ANNOTATION_VERSION:-}" ]]; then
  NORMALIZE_STEM="${EXPORT_STEM:-$ANNOTATION_VERSION}"
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$IATS_HOST" \
    "cd '$IATS_REPO_ROOT' && BIRDS_DATA_ROOT='$IATS_BIRDS_DATA_ROOT' uv run python scripts/export_normalize.py --export-json '$IATS_BIRDS_DATA_ROOT/labelstudio/exports/${NORMALIZE_STEM}.json' --annotation-version '$ANNOTATION_VERSION'"
fi
