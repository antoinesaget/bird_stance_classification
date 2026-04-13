#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_cmd curl docker midclt python3

ENV_FILE_REL="${ENV_FILE:-projects/labelstudio/deploy/env/truenas.env}"
ENV_FILE_PATH="$(resolve_repo_path "$ENV_FILE_REL")"
COMPOSE_FILE_PATH="$(resolve_repo_path "projects/labelstudio/deploy/docker-compose.truenas.yml")"

source_env "$ENV_FILE_PATH"
require_clean_worktree

[[ -d "$LABEL_STUDIO_PGDATA_DIR" ]] || die "Missing LABEL_STUDIO_PGDATA_DIR: $LABEL_STUDIO_PGDATA_DIR"
[[ -d "$LABEL_STUDIO_APP_DATA_DIR" ]] || die "Missing LABEL_STUDIO_APP_DATA_DIR: $LABEL_STUDIO_APP_DATA_DIR"
mkdir -p "$BIRDS_DATA_ROOT/labelstudio/exports"

COMPOSE_JSON="$(render_compose_json "$COMPOSE_FILE_PATH" "$ENV_FILE_PATH")"
APP_ID="${TRUENAS_APP_ID:-bird-stance-classification}"

APP_EXISTS="$(
  midclt call app.query "[[\"name\",\"=\",\"${APP_ID}\"]]" | python3 -c '
import json
import sys
items = json.load(sys.stdin)
print("yes" if items else "no")
'
)"

if [[ "$APP_EXISTS" == "yes" ]]; then
  PAYLOAD="$(
    python3 - "$COMPOSE_JSON" <<'PY'
import json
import sys
print(json.dumps({"custom_compose_config": json.loads(sys.argv[1])}, separators=(",", ":")))
PY
  )"
  ACTION=(app.update "$APP_ID")
else
  PAYLOAD="$(
    python3 - "$APP_ID" "$COMPOSE_JSON" <<'PY'
import json
import sys
print(json.dumps({
    "app_name": sys.argv[1],
    "custom_app": True,
    "custom_compose_config": json.loads(sys.argv[2]),
}, separators=(",", ":")))
PY
  )"
  ACTION=(app.create)
fi

PAYLOAD_B64="$(printf '%s' "$PAYLOAD" | base64 | tr -d '\n')"
JOB_ID="$(midclt call "${ACTION[@]}" "$(echo "$PAYLOAD_B64" | base64 -d)" | tr -d '"')"
[[ -n "$JOB_ID" ]] || die "Failed to start TrueNAS app job"

for _ in $(seq 1 60); do
  JOB_STATE="$(
    midclt call core.get_jobs "[[\"id\",\"=\",${JOB_ID}]]" | python3 -c '
import json
import sys
jobs = json.load(sys.stdin)
print(jobs[0]["state"] if jobs else "")
'
  )"
  case "$JOB_STATE" in
    SUCCESS) break ;;
    FAILED) die "TrueNAS app job ${JOB_ID} failed" ;;
  esac
  sleep 5
done

APP_STATE="$(
  midclt call app.get_instance "$APP_ID" | python3 -c '
import json
import sys
print((json.load(sys.stdin) or {}).get("state", ""))
'
)"
case "$APP_STATE" in
  RUNNING|ACTIVE) ;;
  *) die "Unexpected TrueNAS app state: ${APP_STATE:-unknown}" ;;
esac

for _ in $(seq 1 30); do
  if curl -fsSI "http://127.0.0.1:${TRUENAS_APP_PORT:-30280}/user/login/" >/dev/null; then
    printf '[ops] app=%s state=%s\n' "$APP_ID" "$APP_STATE"
    exit 0
  fi
  sleep 2
done

die "Label Studio login endpoint did not become ready on port ${TRUENAS_APP_PORT:-30280}"
