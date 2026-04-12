#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_cmd python3

ENV_FILE_REL="${ENV_FILE:-deploy/env/iats.env}"
ENV_FILE_PATH="$(resolve_repo_path "$ENV_FILE_REL")"
source_env "$ENV_FILE_PATH"

BIRDS_DATA_ROOT="$(resolve_repo_path "$BIRDS_DATA_ROOT")"
MODEL_B_SERVING_ARTIFACT="${MODEL_B_SERVING_ARTIFACT:-${MODEL_B_SERVING_CHECKPOINT:-data/birds_project/models/attributes/served/model_b/current}}"
MODEL_B_SERVING_ARTIFACT="$(resolve_repo_path "$MODEL_B_SERVING_ARTIFACT")"

if [[ -z "${MODEL_B_SOURCE:-}" ]]; then
  die "MODEL_B_SOURCE is required"
fi
if [[ "$MODEL_B_SOURCE" = /* ]]; then
  MODEL_B_SOURCE_PATH="$MODEL_B_SOURCE"
else
  MODEL_B_SOURCE_PATH="$(resolve_repo_path "$MODEL_B_SOURCE")"
fi
[[ -e "$MODEL_B_SOURCE_PATH" ]] || die "Missing MODEL_B_SOURCE: $MODEL_B_SOURCE_PATH"

if [[ "$(basename "$MODEL_B_SERVING_ARTIFACT")" = "current" ]]; then
  SERVING_DIR="$(dirname "$MODEL_B_SERVING_ARTIFACT")"
else
  SERVING_DIR="$(dirname "$(dirname "$MODEL_B_SERVING_ARTIFACT")")"
fi
mkdir -p "$BIRDS_DATA_ROOT" "$SERVING_DIR"

python3 "$REPO_ROOT/scripts/promote_model.py" \
  --source "$MODEL_B_SOURCE_PATH" \
  --served-dir "$SERVING_DIR" \
  --artifact-name "checkpoint.pt" \
  --label "${PROMOTION_LABEL:-model_b}" \
  --notes "${PROMOTION_NOTES:-Model B promoted via iats_deploy_model_b_remote.sh}"

export MODEL_B_SERVING_ARTIFACT
bash "$REPO_ROOT/scripts/ops/iats_deploy_ml_remote.sh"
