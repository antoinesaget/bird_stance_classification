#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_cmd curl docker python3

ENV_FILE_REL="${ENV_FILE:-deploy/env/iats.env}"
ENV_FILE_PATH="$(resolve_repo_path "$ENV_FILE_REL")"
COMPOSE_FILE_PATH="$(resolve_repo_path "deploy/docker-compose.iats-ml.yml")"

source_env "$ENV_FILE_PATH"
require_clean_worktree

BIRDS_DATA_ROOT="$(resolve_repo_path "$BIRDS_DATA_ROOT")"
MODEL_A_SERVING_WEIGHTS="$(resolve_repo_path "$MODEL_A_SERVING_WEIGHTS")"
MODEL_B_SERVING_ARTIFACT="${MODEL_B_SERVING_ARTIFACT:-${MODEL_B_SERVING_CHECKPOINT:-data/birds_project/models/attributes/served/model_b/current}}"
MODEL_B_SERVING_ARTIFACT="$(resolve_repo_path "$MODEL_B_SERVING_ARTIFACT")"
MODEL_A_BOOTSTRAP_WEIGHTS="${MODEL_A_BOOTSTRAP_WEIGHTS:-}"
if [[ -n "$MODEL_A_BOOTSTRAP_WEIGHTS" ]]; then
  MODEL_A_BOOTSTRAP_WEIGHTS="$(resolve_repo_path "$MODEL_A_BOOTSTRAP_WEIGHTS")"
fi

SERVING_DIR="$(dirname "$(dirname "$MODEL_A_SERVING_WEIGHTS")")"
mkdir -p "$BIRDS_DATA_ROOT" "$SERVING_DIR"

if [[ ! -f "$MODEL_A_SERVING_WEIGHTS" ]]; then
  [[ -n "$MODEL_A_BOOTSTRAP_WEIGHTS" && -f "$MODEL_A_BOOTSTRAP_WEIGHTS" ]] || die "MODEL_A_SERVING_WEIGHTS is missing and MODEL_A_BOOTSTRAP_WEIGHTS is unavailable"
  log "Bootstrapping served detector weights from $MODEL_A_BOOTSTRAP_WEIGHTS"
  python3 "$REPO_ROOT/scripts/promote_model.py" \
    --source "$MODEL_A_BOOTSTRAP_WEIGHTS" \
    --served-dir "$SERVING_DIR" \
    --label bootstrap \
    --notes "Initial promotion created by iats_deploy_ml_remote.sh"
fi

if [[ "$MODEL_B_SERVING_ARTIFACT" == "$BIRDS_DATA_ROOT" ]]; then
  MODEL_B_CHECKPOINT="/data/birds_project"
elif [[ "$MODEL_B_SERVING_ARTIFACT" == "$BIRDS_DATA_ROOT/"* ]]; then
  MODEL_B_CHECKPOINT="/data/birds_project/${MODEL_B_SERVING_ARTIFACT#"$BIRDS_DATA_ROOT"/}"
else
  die "MODEL_B_SERVING_ARTIFACT must live under BIRDS_DATA_ROOT: $MODEL_B_SERVING_ARTIFACT"
fi

export BIRDS_DATA_ROOT MODEL_A_SERVING_WEIGHTS MODEL_A_BOOTSTRAP_WEIGHTS MODEL_B_SERVING_ARTIFACT MODEL_B_CHECKPOINT

docker compose --env-file "$ENV_FILE_PATH" -f "$COMPOSE_FILE_PATH" config >/dev/null
docker compose --env-file "$ENV_FILE_PATH" -f "$COMPOSE_FILE_PATH" up -d --build

if [[ "${IATS_STOP_LEGACY_UI:-0}" == "1" ]]; then
  log "Stopping legacy iats UI containers"
  docker rm -f birds-label-studio birds-postgres >/dev/null 2>&1 || true
fi

HEALTH_JSON=""
for _ in $(seq 1 30); do
  if HEALTH_JSON="$(curl -fsS "http://127.0.0.1:${ML_BACKEND_PORT:-9090}/health")"; then
    break
  fi
  sleep 2
done

[[ -n "$HEALTH_JSON" ]] || die "ML backend health endpoint did not become ready on port ${ML_BACKEND_PORT:-9090}"
printf '%s\n' "$HEALTH_JSON"
HEALTH_JSON="$HEALTH_JSON" REQUIRE_NON_CPU_DEVICE="${REQUIRE_NON_CPU_DEVICE:-1}" python3 - <<'PY'
import json
import os

payload = json.loads(os.environ["HEALTH_JSON"])
if not payload.get("model_a_loaded"):
    raise SystemExit("Model A did not load")
if os.environ.get("REQUIRE_NON_CPU_DEVICE", "1") not in {"0", "false", "False"}:
    if payload.get("model_a_device") in {None, "cpu"}:
        raise SystemExit(f"Expected non-CPU device, got {payload.get(\"model_a_device\")!r}")
model_b_checkpoint = os.environ.get("MODEL_B_SERVING_CHECKPOINT")
if not model_b_checkpoint:
    model_b_checkpoint = os.environ.get("MODEL_B_SERVING_ARTIFACT")
if model_b_checkpoint and os.path.exists(model_b_checkpoint):
    if not payload.get("model_b_loaded"):
        raise SystemExit("Model B checkpoint exists but backend did not load it")
PY

docker inspect birds-ml-backend --format '{{json .HostConfig.DeviceRequests}}'
