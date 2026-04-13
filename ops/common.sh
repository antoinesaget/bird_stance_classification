#!/usr/bin/env bash
# Purpose: Provide shared shell helpers used by every operational script in the repo.

OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$OPS_DIR/.." && pwd)"

log() {
  printf '[ops] %s\n' "$*"
}

die() {
  printf '[ops] ERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  local cmd
  for cmd in "$@"; do
    command -v "$cmd" >/dev/null 2>&1 || die "Missing required command: $cmd"
  done
}

repo_python_bin() {
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    printf '%s\n' "$REPO_ROOT/.venv/bin/python"
    return
  fi
  command -v python3 >/dev/null 2>&1 || die "Missing required command: python3"
  command -v python3
}

repo_pythonpath() {
  local parts=(
    "$REPO_ROOT/workspace_bootstrap/src"
    "$REPO_ROOT/ops/src"
    "$REPO_ROOT/shared/birdsys_core/src"
    "$REPO_ROOT/projects/labelstudio/src"
    "$REPO_ROOT/projects/datasets/src"
    "$REPO_ROOT/projects/ml_backend/src"
    "$REPO_ROOT/projects/ml_experiments/src"
  )
  local joined
  IFS=: joined="${parts[*]}"
  printf '%s\n' "$joined"
}

run_birdsys() {
  local python_bin="${PYTHON_BIN:-$(repo_python_bin)}"
  PYTHONPATH="$(repo_pythonpath)${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" -m birdsys_workspace.cli "$@"
}

source_env() {
  local env_file="$1"
  [[ -f "$env_file" ]] || die "Missing env file: $env_file"
  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a
}

resolve_repo_path() {
  local raw="${1:-}"
  [[ -n "$raw" ]] || die "Expected a non-empty path"
  if [[ "$raw" = /* ]]; then
    printf '%s\n' "$raw"
    return
  fi
  printf '%s\n' "$REPO_ROOT/$raw"
}

require_clean_worktree() {
  local status
  status="$(git -C "$REPO_ROOT" status --porcelain)"
  [[ -z "$status" ]] || {
    printf '%s\n' "$status" >&2
    die "Worktree must be clean in $REPO_ROOT"
  }
}

git_pull_ff_only() {
  local branch="${1:-${DEPLOY_BRANCH:-main}}"
  require_cmd git
  git -C "$REPO_ROOT" fetch origin "$branch"
  if git -C "$REPO_ROOT" show-ref --verify --quiet "refs/heads/$branch"; then
    git -C "$REPO_ROOT" checkout "$branch"
  else
    git -C "$REPO_ROOT" checkout -B "$branch" "origin/$branch"
  fi
  git -C "$REPO_ROOT" pull --ff-only origin "$branch"
}

render_compose_json() {
  local compose_file="$1"
  local env_file="$2"
  require_cmd docker python3
  REPO_ROOT="$REPO_ROOT" docker compose --env-file "$env_file" -f "$compose_file" config --format json | python3 -c '
import json, sys
data = json.load(sys.stdin)
out = {}
for key in ("services", "volumes", "networks"):
    value = data.get(key)
    if value:
        out[key] = value
print(json.dumps(out, separators=(",", ":")))
'
}
