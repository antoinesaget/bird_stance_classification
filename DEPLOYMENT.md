# Deployment Runbook

This repo has three deployment surfaces:

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

