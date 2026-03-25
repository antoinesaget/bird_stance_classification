#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_cmd rsync ssh

TRUENAS_HOST="${TRUENAS_HOST:-truenas}"
IATS_HOST="${IATS_HOST:-iats}"
TRUENAS_BIRDS_DATA_ROOT="${TRUENAS_BIRDS_DATA_ROOT:-/mnt/tank/media/birds_project}"
IATS_BIRDS_DATA_ROOT="${IATS_BIRDS_DATA_ROOT:-/home/antoine/bird_stance_classification/data/birds_project}"

SYNC_PATHS=(
  raw_images
  metadata
  labelstudio/exports
)

ssh -o BatchMode=yes -o ConnectTimeout=15 "$IATS_HOST" "mkdir -p '$IATS_BIRDS_DATA_ROOT/raw_images' '$IATS_BIRDS_DATA_ROOT/metadata' '$IATS_BIRDS_DATA_ROOT/labelstudio/exports'"

for rel_path in "${SYNC_PATHS[@]}"; do
  log "Syncing ${rel_path} from ${TRUENAS_HOST} to ${IATS_HOST}"
  rsync -az --delete \
    "${TRUENAS_HOST}:${TRUENAS_BIRDS_DATA_ROOT}/${rel_path}/" \
    "${IATS_HOST}:${IATS_BIRDS_DATA_ROOT}/${rel_path}/"
done
