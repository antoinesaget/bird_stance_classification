# Deployment Runbook

This repo has three deployment surfaces:

- `ops/compose/docker-compose.local.yml`
- `projects/ml_backend/deploy/docker-compose.iats-ml.yml`
- `projects/labelstudio/deploy/docker-compose.truenas.yml`

The live branch is `main`. The Makefile already defaults `DEPLOY_BRANCH` to `main`, so only pass it explicitly for temporary branch rollouts.

## Host Roles

- `iats`
  - repo checkout: `/home/antoine/bird_stance_classification`
  - same full tracked checkout as local
  - training and experiment host
  - live ML backend host
- TrueNAS
  - repo checkout: `/mnt/apps/code/bird_stance_classification`
  - minimal deployment checkout: `ops`, `projects/labelstudio`, `shared`, `workspace_bootstrap`
  - live Label Studio/Postgres/public UI host
  - canonical storage host for `birds_project` and `lines_project`
- Local
  - orchestration and smoke-test host

## Required Env Files

- local: `ops/env/local.env`
- `iats`: `projects/ml_backend/deploy/env/iats.env`
- TrueNAS: `projects/labelstudio/deploy/env/truenas.env`

Critical values:

- `projects/ml_backend/deploy/env/iats.env`
  - `BIRDS_DATA_ROOT=/data/birds_project`
  - `LINES_DATA_ROOT=/data/lines_project`
  - `MODEL_A_DEVICE=0`
  - `MODEL_A_BOOTSTRAP_WEIGHTS`
  - `MODEL_A_SERVING_WEIGHTS`
  - `MODEL_B_SERVING_ARTIFACT=/data/birds_project/models/attributes/served/model_b/current`
- `projects/labelstudio/deploy/env/truenas.env`
  - `TRUENAS_APP_ID=bird-stance-classification`
  - `BIRDS_DATA_ROOT=/mnt/tank/media/birds_project`
  - `LINES_DATA_ROOT=/mnt/tank/media/lines_project`
  - `LABEL_STUDIO_URL=http://127.0.0.1:30280`
  - `LABEL_STUDIO_API_TOKEN`

## Clean Pull / Bootstrap

```bash
make iats-pull
make truenas-pull
```

Behavior:

- clone if the remote checkout does not exist
- checkout `DEPLOY_BRANCH`
- pull `--ff-only`
- fail if the remote worktree is dirty

## Deploy TrueNAS UI

```bash
make truenas-deploy-ui
```

This re-renders `projects/labelstudio/deploy/docker-compose.truenas.yml` and updates the existing `bird-stance-classification` app while preserving persistent state mounts.

## Sync Data And Exports To `iats`

Canonical sync:

```bash
make iats-sync-data
```

Export-only sync:

```bash
make iats-import-exports PROJECT_ID=4 EXPORT_NAME=ann_v002_legacy
```

The sync path is intentionally one-way for canonical inputs:

- TrueNAS owns `raw_images/`, `metadata/`, and `labelstudio/exports/`
- `iats` owns derived datasets, training outputs, and served model slots

## Train And Deploy Model B On `iats`

Cross-validation:

```bash
make iats-train-attributes-cv DATASET_DIR=/data/birds_project/derived/datasets/ds_v001
```

Final training:

```bash
make iats-train-attributes-final DATASET_DIR=/data/birds_project/derived/datasets/ds_v001
```

Promote and deploy the final checkpoint:

```bash
make iats-deploy-model-b MODEL_B_SOURCE=/data/birds_project/models/attributes/convnextv2s_v001/checkpoint.pt PROMOTION_LABEL=ann_v002_legacy
```

Deploy or re-deploy the ML backend container itself:

```bash
make iats-deploy-ml
```

## `lines_project` Batch Flow On TrueNAS

Prepare the `q60` mirror and import bundle:

```bash
make truenas-prepare-lines-batch LINES_BATCH_NAME=lines_bw_stilts_5k_seed_20260325_q60
```

Import the generated task bundle into project `7`:

```bash
make truenas-import-lines-batch LINES_PROJECT_ID=7 LINES_BATCH_NAME=lines_bw_stilts_5k_seed_20260325_q60
```

Persist predictions so they do not have to be generated on the fly:

```bash
make truenas-prefill-lines-predictions LINES_PROJECT_ID=7 LINES_BATCH_NAME=lines_bw_stilts_5k_seed_20260325_q60 LINES_ONLY_MISSING=1
```

## Verification

Cross-host smoke:

```bash
make smoke-remote
```

Expected state:

- TrueNAS app is healthy and [birds.ashs.live](https://birds.ashs.live/user/login/) loads
- `iats` ML backend `/health` returns `status=UP`
- project `7` can reach the ML backend over `http://192.168.0.42:9090`
- served model paths on `iats` resolve under `/data/birds_project/models`
