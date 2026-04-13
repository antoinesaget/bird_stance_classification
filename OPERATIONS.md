# BirdSys Operations

This is the single canonical operations document for the current cleaned repo. It merges the old environment and deployment notes and only describes the surfaces that still exist.

## Repo Scope

- Active branch for deployments: `main`
- Active deployment surfaces:
  - `projects/ml_backend/deploy/docker-compose.iats-ml.yml`
  - `projects/labelstudio/deploy/docker-compose.truenas.yml`
- Removed from the repo:
  - root `Makefile`
  - root orchestration layer under `ops/`
  - repo-wide deploy/bootstrap wrappers

## Host Roles

- Local
  - source-of-truth engineering checkout
  - repo maintenance
  - remote orchestration when needed
- `iats`
  - full engineering checkout matching local
  - training and experiment host
  - live ML backend host
- TrueNAS
  - minimal deployment checkout for Label Studio
  - live Label Studio and Postgres host
  - canonical storage host for `birds_project` and `lines_project`

## Canonical Paths

### Local

- Repo root: `/Users/antoine/truenas_migration/bird_stance_classification`
- Archive root for local cleanup and old snapshots: `/Users/antoine/truenas_migration/_archives/bird_stance_classification`

### `iats`

- Repo checkout: `/home/antoine/bird_stance_classification`
- Bird data root: `/data/birds_project`
- Lines data root: `/data/lines_project`
- Served detector slot: `/data/birds_project/models/detector/served/model_a/current/weights.pt`
- Served attribute artifact slot: `/data/birds_project/models/attributes/served/model_b/current`

### TrueNAS

- Repo checkout: `/mnt/apps/code/bird_stance_classification`
- Stable app id: `bird-stance-classification`
- Bird data root: `/mnt/tank/media/birds_project`
- Lines data root: `/mnt/tank/media/lines_project`
- Label Studio imports root: `/mnt/tank/media/lines_project/labelstudio/imports`
- Lines compressed mirror root: `/mnt/tank/media/lines_project/labelstudio/images_compressed/lines_bw_stilts_q60`

## Data Ownership

TrueNAS is authoritative for:

- `birds_project/raw_images`
- `birds_project/metadata`
- `birds_project/labelstudio/exports`
- `lines_project/labelstudio/imports`
- `lines_project/labelstudio/images_compressed`

`iats` owns:

- normalized datasets used for training
- derived training artifacts
- served model releases and current live model slots

## Deployment Files

### ML Backend on `iats`

- Compose file: `projects/ml_backend/deploy/docker-compose.iats-ml.yml`
- Example env file: `projects/ml_backend/deploy/env/iats.env.example`
- Container name: `birds-ml-backend`
- Port mapping default: `9090:9090`
- Mounted data roots:
  - `${BIRDS_DATA_ROOT} -> /data/birds_project` read-only
  - `${LINES_DATA_ROOT} -> /data/lines_project` read-only

Expected key environment values:

- `BIRDS_DATA_ROOT=/data/birds_project`
- `LINES_DATA_ROOT=/data/lines_project`
- `MODEL_A_DEVICE=0`
- `MODEL_A_IMGSZ=1280`
- `MODEL_A_MAX_DET=300`
- `MODEL_A_CONF=0.25`
- `MODEL_A_IOU=0.45`
- `MODEL_B_CHECKPOINT=/data/birds_project/models/attributes/served/model_b/current`
- `BIRDS_LOG_LEVEL=INFO`
- `ML_BACKEND_PORT=9090`

### Label Studio on TrueNAS

- Compose file: `projects/labelstudio/deploy/docker-compose.truenas.yml`
- Example env file: `projects/labelstudio/deploy/env/truenas.env.example`
- Containers:
  - `birds-postgres`
  - `birds-label-studio`
- Port mapping default: `30280:8080`
- Mounted data roots:
  - `${BIRDS_DATA_ROOT} -> /data/birds_project`
  - `${LINES_DATA_ROOT} -> /data/lines_project`

Expected key environment values:

- `TRUENAS_APP_ID=bird-stance-classification`
- `TRUENAS_APP_PORT=30280`
- `BIRDS_DATA_ROOT=/mnt/tank/media/birds_project`
- `LINES_DATA_ROOT=/mnt/tank/media/lines_project`
- `LABEL_STUDIO_URL=https://birds.ashs.live`
- `LABEL_STUDIO_HOST=https://birds.ashs.live`
- `LABEL_STUDIO_API_TOKEN`
- `LABEL_STUDIO_ML_BACKEND_URL=http://192.168.0.42:9090`
- `LABEL_STUDIO_PGDATA_DIR=/mnt/apps/configs/bird-stance-classification/postgres`
- `LABEL_STUDIO_APP_DATA_DIR=/mnt/apps/configs/bird-stance-classification/labelstudio_data`

## Live Expectations

### ML Backend

- Runs on `iats`
- GPU-backed
- `/health` should report:
  - `status=UP`
  - `model_a_loaded=true`
  - `model_b_loaded=true`
  - `model_b_schema_version=annotation_schema_v2`

### Label Studio

- Public URL: [birds.ashs.live](https://birds.ashs.live)
- Stable state lives on TrueNAS only
- Expected ML backend target: `http://192.168.0.42:9090`

## Current Repo Reality

- The deployment files above still exist and appear to be the intended production surfaces.
- The root repo no longer contains a general deployment wrapper.
- Most Python entrypoints outside the Label Studio API helpers are currently blocked because `shared/birdsys_core/src/birdsys/core/__init__.py` still imports deleted modules.
- Operational docs should treat this repo as a reduced codebase under repair, not as a fully working monorepo.
