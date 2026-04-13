#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_cmd docker

ACTION="${1:?usage: local_compose.sh <config|up|down|ps|stop-ml>}"
ENV_FILE_REL="${ENV_FILE:-ops/env/local.env}"
ENV_FILE_PATH="$(resolve_repo_path "$ENV_FILE_REL")"
COMPOSE_FILE_PATH="$(resolve_repo_path "ops/compose/docker-compose.local.yml")"

source_env "$ENV_FILE_PATH"

BIRDS_DATA_ROOT="$(resolve_repo_path "$BIRDS_DATA_ROOT")"
MODEL_A_SERVING_WEIGHTS="$(resolve_repo_path "$MODEL_A_SERVING_WEIGHTS")"
LABEL_STUDIO_PGDATA_DIR="$(resolve_repo_path "$LABEL_STUDIO_PGDATA_DIR")"
LABEL_STUDIO_APP_DATA_DIR="$(resolve_repo_path "$LABEL_STUDIO_APP_DATA_DIR")"

mkdir -p "$BIRDS_DATA_ROOT" "$LABEL_STUDIO_PGDATA_DIR" "$LABEL_STUDIO_APP_DATA_DIR" "$(dirname "$MODEL_A_SERVING_WEIGHTS")"

export BIRDS_DATA_ROOT MODEL_A_SERVING_WEIGHTS LABEL_STUDIO_PGDATA_DIR LABEL_STUDIO_APP_DATA_DIR

case "$ACTION" in
  config)
    docker compose --env-file "$ENV_FILE_PATH" -f "$COMPOSE_FILE_PATH" config
    ;;
  up)
    docker compose --env-file "$ENV_FILE_PATH" -f "$COMPOSE_FILE_PATH" up -d --build
    ;;
  down)
    docker compose --env-file "$ENV_FILE_PATH" -f "$COMPOSE_FILE_PATH" down
    ;;
  ps)
    docker compose --env-file "$ENV_FILE_PATH" -f "$COMPOSE_FILE_PATH" ps
    ;;
  stop-ml)
    docker compose --env-file "$ENV_FILE_PATH" -f "$COMPOSE_FILE_PATH" stop ml-backend
    ;;
  *)
    die "Unsupported local compose action: $ACTION"
    ;;
esac
