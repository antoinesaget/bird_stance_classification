#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_cmd curl python3 ssh

IATS_HOST="${IATS_HOST:-iats}"
TRUENAS_HOST="${TRUENAS_HOST:-truenas}"
APP_ID="${TRUENAS_APP_ID:-bird-stance-classification}"

IATS_IP="$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$IATS_HOST" "hostname -I | awk '{print \$1}'")"
ML_HEALTH="$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$IATS_HOST" "curl -fsS http://127.0.0.1:9090/health")"
printf '%s\n' "$ML_HEALTH"
printf '%s\n' "$ML_HEALTH" | python3 -c '
import json
import sys
payload = json.load(sys.stdin)
if payload.get("status") != "UP":
    raise SystemExit("ML backend health did not report UP")
'

APP_STATE="$(
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$TRUENAS_HOST" "midclt call app.get_instance '$APP_ID'" | python3 -c '
import json
import sys
print((json.load(sys.stdin) or {}).get("state", ""))
'
)"
printf '[ops] truenas app=%s state=%s\n' "$APP_ID" "$APP_STATE"

curl -fsSI https://birds.ashs.live/user/login/ >/dev/null
ssh -o BatchMode=yes -o ConnectTimeout=15 "$TRUENAS_HOST" "curl -fsS http://${IATS_IP}:9090/health" >/dev/null
