# BirdSys Operations

This is the single canonical operations document for the current cleaned repo. It merges the old environment and deployment notes and only describes the surfaces that still exist.

## Repo Scope

- Active branch for deployments: `main`
- Active deployment surfaces:
  - `projects/ml_backend/deploy/docker-compose.ai-ml.yml`
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
- `ai`
  - full engineering checkout matching local
  - headless training and experiment host
  - live ML backend host
- TrueNAS
  - minimal deployment checkout for Label Studio
  - live Label Studio and Postgres host
  - canonical storage host for `/mnt/tank/media/birds`

## Canonical Paths

### Local

- Repo root: `/Users/antoine/truenas_migration/bird_stance_classification`
- Archive root for local cleanup and old snapshots: `/Users/antoine/truenas_migration/_archives/bird_stance_classification`

### `ai`

- Repo checkout: `/home/antoine/bird_stance_classification`
- Bird data home: `/data/birds`
- Active species root: `/data/birds/black_winged_stilt`
- Legacy species root: `/data/birds/old_specie`
- Served detector slot: `/data/birds/black_winged_stilt/models/detector/served/model_a/current/weights.pt`
- Served attribute artifact slot: `/data/birds/black_winged_stilt/models/attributes/served/model_b/current`

### TrueNAS

- Repo checkout: `/mnt/apps/code/bird_stance_classification`
- Stable app id: `bird-stance-classification`
- Bird data home: `/mnt/tank/media/birds`
- Active species root: `/mnt/tank/media/birds/black_winged_stilt`
- Legacy species root: `/mnt/tank/media/birds/old_specie`
- Label Studio imports root: `/mnt/tank/media/birds/black_winged_stilt/labelstudio/imports`
- Compressed mirror root: `/mnt/tank/media/birds/black_winged_stilt/labelstudio/images_compressed/q60`

## Data Ownership

TrueNAS is authoritative for:

- `birds/<species_slug>/originals`
- `birds/<species_slug>/metadata`
- `birds/<species_slug>/labelstudio/imports`
- `birds/<species_slug>/labelstudio/images_compressed`
- `birds/<species_slug>/labelstudio/exports`

`ai` owns:

- normalized datasets used for training
- derived training artifacts
- served model releases and current live model slots

## Deployment Files

### ML Backend on `ai`

- Compose file: `projects/ml_backend/deploy/docker-compose.ai-ml.yml`
- Example env file: `projects/ml_backend/deploy/env/ai.env.example`
- Container name: `birds-ml-backend`
- Port mapping default: `9090:9090`
- Mounted data roots:
  - `${BIRD_DATA_HOME} -> /data/birds` read-only

Expected key environment values:

- `BIRD_DATA_HOME=/data/birds`
- `BIRD_SPECIES_SLUG=black_winged_stilt`
- `MODEL_A_DEVICE=0`
- `MODEL_A_IMGSZ=1280`
- `MODEL_A_MAX_DET=300`
- `MODEL_A_CONF=0.25`
- `MODEL_A_IOU=0.45`
- `MODEL_B_CHECKPOINT=/data/birds/black_winged_stilt/models/attributes/served/model_b/current`
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
  - `${BIRD_DATA_HOME} -> /data/birds`

Expected key environment values:

- `TRUENAS_APP_ID=bird-stance-classification`
- `TRUENAS_APP_PORT=30280`
- `BIRD_DATA_HOME=/mnt/tank/media/birds`
- `BIRD_SPECIES_SLUG=black_winged_stilt`
- `LABEL_STUDIO_URL=https://birds.ashs.live`
- `LABEL_STUDIO_HOST=https://birds.ashs.live`
- `LABEL_STUDIO_API_TOKEN`
- `LABEL_STUDIO_ML_BACKEND_URL=http://ai.tahr-hoki.ts.net:9090`
- `LABEL_STUDIO_PGDATA_DIR=/mnt/apps/configs/bird-stance-classification/postgres`
- `LABEL_STUDIO_APP_DATA_DIR=/mnt/apps/configs/bird-stance-classification/labelstudio_data`

## Live Expectations

### ML Backend

- Runs on `ai`
- GPU-backed
- `/health` should report:
  - `status=UP`
  - `model_a_loaded=true`
  - `model_b_loaded=true`
  - `model_b_schema_version=annotation_schema_v2`

### Label Studio

- Public URL: [birds.ashs.live](https://birds.ashs.live)
- Stable state lives on TrueNAS only
- Expected ML backend target: `http://ai.tahr-hoki.ts.net:9090`

## Current Repo Reality

- The deployment files above still exist and appear to be the intended production surfaces.
- The root repo no longer contains a general deployment wrapper.
- Most Python entrypoints outside the Label Studio API helpers are currently blocked because `shared/birdsys_core/src/birdsys/core/__init__.py` still imports deleted modules.
- Operational docs should treat this repo as a reduced codebase under repair, not as a fully working monorepo.
