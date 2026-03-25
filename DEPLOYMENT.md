# Deployment Runbook

This repository now uses three explicit deployment surfaces:

- `deploy/docker-compose.local.yml`
- `deploy/docker-compose.iats-ml.yml`
- `deploy/docker-compose.truenas.yml`

## 1) Host Roles

### `iats`

- Primary engineering checkout
- Training and experiment host
- Production ML backend host
- Receives exports and training inputs from TrueNAS

### TrueNAS

- Stable Label Studio/Postgres/public UI host
- Canonical `birds_project` storage host
- Source of annotation exports

### Local

- Optional local debug stack
- Low-priority orchestration only

## 2) Required Env Files

Create these untracked files from the examples:

```bash
cp deploy/env/local.env.example deploy/env/local.env
```

On `iats`:

```bash
cp deploy/env/iats.env.example deploy/env/iats.env
```

On TrueNAS:

```bash
cp deploy/env/truenas.env.example deploy/env/truenas.env
```

Minimum important values:

- `deploy/env/iats.env`
  - `BIRDS_DATA_ROOT`
  - `MODEL_A_BOOTSTRAP_WEIGHTS`
  - `MODEL_A_SERVING_WEIGHTS`
  - `MODEL_A_DEVICE=0`
- `deploy/env/truenas.env`
  - `TRUENAS_APP_ID=bird-stance-classification`
  - `BIRDS_DATA_ROOT=/mnt/tank/media/birds_project`
  - `LABEL_STUDIO_HOST=https://birds.ashs.live`
  - `LABEL_STUDIO_API_TOKEN`
  - `LABEL_STUDIO_PGDATA_DIR`
  - `LABEL_STUDIO_APP_DATA_DIR`

## 3) Clean Pull / Bootstrap

Both managed pull commands fail on dirty worktrees.

```bash
make iats-pull
make truenas-pull
```

Behavior:

- If the remote checkout does not exist, it is cloned first.
- The checkout is moved to `DEPLOY_BRANCH` (default `main`).
- The pull is `--ff-only`.

Fetch/clone uses the public HTTPS repo URL by default. This avoids the current GitHub SSH auth gap on `iats`.

## 4) Stable Frontend Deploy On TrueNAS

Pull the repo, then redeploy the existing custom app from the repo-owned compose file:

```bash
make truenas-pull
make truenas-deploy-ui
```

What the deploy does:

- renders `deploy/docker-compose.truenas.yml`
- updates or creates the `bird-stance-classification` custom app through `midclt app.update/app.create`
- keeps Postgres and Label Studio state mounted from the existing TrueNAS config paths
- mounts `deploy/overrides/localfiles_views.py` directly from the repo checkout instead of maintaining an ad hoc copy

## 5) Annotation Export Flow

Export project annotations from the stable TrueNAS UI into the canonical dataset tree:

```bash
make truenas-export-annotations PROJECT_ID=4 ANNOTATION_VERSION=ann_v001
```

This writes:

- `${BIRDS_DATA_ROOT}/labelstudio/exports/ann_v001.json`
- `${BIRDS_DATA_ROOT}/labelstudio/exports/ann_v001-info.json`

The export uses the Label Studio API token from `deploy/env/truenas.env`.

## 6) Training Input Sync To `iats`

Full canonical training-input sync:

```bash
make iats-sync-data
```

This syncs only the canonical input subsets:

- `raw_images/`
- `metadata/`
- `labelstudio/exports/`

It intentionally does not overwrite `iats` model experiment outputs or derived training artifacts.

Export-only sync for faster iteration:

```bash
make iats-import-exports PROJECT_ID=4 ANNOTATION_VERSION=ann_v001
```

This can:

- trigger the export on TrueNAS
- copy the export JSON and metadata to `iats`
- optionally run normalization on `iats`

## 7) Training And Promotion On `iats`

Example attribute training:

```bash
make iats-train TRAIN_PIPELINE=attributes DATASET_VERSION=ds_v001
```

Example image-status training:

```bash
make iats-train TRAIN_PIPELINE=image-status ANNOTATION_VERSION=ann_v001
```

Custom training command:

```bash
make iats-train TRAIN_CMD='uv run python path/to/custom_train.py --arg value'
```

Promote a detector artifact into the served slot:

```bash
make iats-promote-model PROMOTION_SOURCE=data/birds_project/models/detector/experiments/run_001/weights.pt
```

Promotion writes:

- `.../served/model_a/releases/<timestamp>/weights.pt`
- `.../served/model_a/releases/<timestamp>/promotion.json`
- `.../served/model_a/current/weights.pt`
- `.../served/model_a/current/promotion.json`

The promotion target then recreates the ML container so the live backend reloads the newly promoted weights.

## 8) ML Backend Deploy On `iats`

```bash
make iats-deploy-ml
```

What it does:

- bootstraps the served weights slot from `MODEL_A_BOOTSTRAP_WEIGHTS` if needed
- renders `deploy/docker-compose.iats-ml.yml`
- deploys the GPU-backed ML container
- validates `/health`
- fails if `REQUIRE_NON_CPU_DEVICE=1` and the backend still reports `cpu`

Optional cleanup of the legacy `iats` UI containers:

```bash
make iats-deploy-ml IATS_STOP_LEGACY_UI=1
```

## 9) Verification

Local smoke tests:

```bash
uv run pytest -q tests/smoke
```

Cross-host smoke checks:

```bash
make smoke-remote
```

Expected:

- TrueNAS app state is `RUNNING` or `ACTIVE`
- `https://birds.ashs.live/user/login/` returns successfully
- `iats` ML backend returns `status=UP`
- TrueNAS can reach the `iats` ML backend over the LAN
