#!/usr/bin/env bash
# Purpose: Promote a supplied model artifact into the served slot on iats without retraining it.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_cmd curl docker python3

ENV_FILE_REL="${ENV_FILE:-projects/ml_backend/deploy/env/iats.env}"
ENV_FILE_PATH="$(resolve_repo_path "$ENV_FILE_REL")"
COMPOSE_FILE_PATH="$(resolve_repo_path "projects/ml_backend/deploy/docker-compose.iats-ml.yml")"
source_env "$ENV_FILE_PATH"

[[ -n "${PROMOTION_SOURCE:-}" ]] || die "PROMOTION_SOURCE is required"

PROMOTION_SOURCE="$(resolve_repo_path "$PROMOTION_SOURCE")"
MODEL_A_SERVING_WEIGHTS="$(resolve_repo_path "$MODEL_A_SERVING_WEIGHTS")"
SERVING_DIR="$(dirname "$(dirname "$MODEL_A_SERVING_WEIGHTS")")"

run_birdsys backend promote-model \
  --source "$PROMOTION_SOURCE" \
  --served-dir "$SERVING_DIR" \
  --label "${PROMOTION_LABEL:-manual}" \
  --notes "${PROMOTION_NOTES:-}"

export BIRDS_DATA_ROOT MODEL_A_SERVING_WEIGHTS
docker compose --env-file "$ENV_FILE_PATH" -f "$COMPOSE_FILE_PATH" up -d --force-recreate ml-backend
curl -fsS "http://127.0.0.1:${ML_BACKEND_PORT:-9090}/health"
