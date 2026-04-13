#!/usr/bin/env bash
# Purpose: Sync canonical bird inputs from TrueNAS into the iats engineering checkout.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_cmd ssh tar

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
  dest_path="$IATS_BIRDS_DATA_ROOT/$rel_path"
  parent_dir="$(dirname "$dest_path")"
  dest_name="$(basename "$dest_path")"
  stage_path="$parent_dir/.${dest_name}.incoming"
  backup_path="$parent_dir/.${dest_name}.previous"

  log "Syncing ${rel_path} from ${TRUENAS_HOST} to ${IATS_HOST}"
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$IATS_HOST" \
    "mkdir -p '$parent_dir' && rm -rf '$stage_path' '$backup_path' && mkdir -p '$stage_path'"
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$TRUENAS_HOST" \
    "tar -C '$TRUENAS_BIRDS_DATA_ROOT/$rel_path' -cpf - ." \
    | ssh -o BatchMode=yes -o ConnectTimeout=15 "$IATS_HOST" \
      "tar -C '$stage_path' -xpf -"
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$IATS_HOST" \
    "rm -rf '$backup_path' && if [ -e '$dest_path' ]; then mv '$dest_path' '$backup_path'; fi && mv '$stage_path' '$dest_path' && rm -rf '$backup_path'"
done
