#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [[ -n "${REMOTE_REPO_URL:-}" ]]; then
  git -C "$REPO_ROOT" remote set-url origin "$REMOTE_REPO_URL"
fi
if [[ -n "${REMOTE_PUSH_URL:-}" ]]; then
  git -C "$REPO_ROOT" remote set-url --push origin "$REMOTE_PUSH_URL"
fi

require_clean_worktree
git_pull_ff_only "${DEPLOY_BRANCH:-main}"
git -C "$REPO_ROOT" log -1 --oneline
