#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_cmd git ssh

HOST="${1:?usage: remote_repo_exec.sh <host> <remote_repo_root> <remote_script> [args...]}"
REMOTE_REPO_ROOT="${2:?usage: remote_repo_exec.sh <host> <remote_repo_root> <remote_script> [args...]}"
REMOTE_SCRIPT="${3:?usage: remote_repo_exec.sh <host> <remote_repo_root> <remote_script> [args...]}"
shift 3

REMOTE_REPO_URL="${REMOTE_REPO_URL:-$(git -C "$REPO_ROOT" config --get remote.origin.url)}"
[[ -n "$REMOTE_REPO_URL" ]] || die "REMOTE_REPO_URL is required"

PASS_ENV_VARS=(
  DEPLOY_BRANCH
  REMOTE_REPO_URL
  REMOTE_PUSH_URL
  ENV_FILE
  TRAIN_PIPELINE
  DATASET_DIR
  DATASET_VERSION
  ANNOTATION_VERSION
  TRAIN_SMOKE
  TRAIN_ARGS
  TRAIN_CMD
  PROJECT_ID
  EXPORT_NAME
  EXPORT_TITLE
  DOWNLOAD_ALL_TASKS
  NORMALIZE_ON_IATS
  PROMOTION_SOURCE
  PROMOTION_LABEL
  PROMOTION_NOTES
  IATS_STOP_LEGACY_UI
  REQUIRE_NON_CPU_DEVICE
)

printf -v REMOTE_ROOT_Q '%q' "$REMOTE_REPO_ROOT"
printf -v REMOTE_PARENT_Q '%q' "$(dirname "$REMOTE_REPO_ROOT")"
printf -v REMOTE_REPO_URL_Q '%q' "$REMOTE_REPO_URL"
printf -v REMOTE_SCRIPT_Q '%q' "$REMOTE_SCRIPT"

REMOTE_CMD="set -euo pipefail;"
REMOTE_CMD+=" if [[ ! -d ${REMOTE_ROOT_Q}/.git ]]; then mkdir -p ${REMOTE_PARENT_Q}; git clone ${REMOTE_REPO_URL_Q} ${REMOTE_ROOT_Q}; fi;"
REMOTE_CMD+=" cd ${REMOTE_ROOT_Q};"
REMOTE_CMD+=" export REMOTE_REPO_ROOT=${REMOTE_ROOT_Q};"

for name in "${PASS_ENV_VARS[@]}"; do
  if [[ ${!name+x} == x ]]; then
    printf -v VALUE_Q '%q' "${!name}"
    REMOTE_CMD+=" export ${name}=${VALUE_Q};"
  fi
done

REMOTE_CMD+=" bash ${REMOTE_SCRIPT_Q}"
for arg in "$@"; do
  printf -v ARG_Q '%q' "$arg"
  REMOTE_CMD+=" ${ARG_Q}"
done

ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" "$REMOTE_CMD"
