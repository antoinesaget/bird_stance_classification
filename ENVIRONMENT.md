# BirdSys Environment Contract

## Runtime Roles

- Local
  - repo maintenance
  - light smoke tests
  - remote orchestration
- `iats`
  - full engineering checkout matching local
  - training and experiment host
  - live ML backend host
- TrueNAS
  - minimal deployment checkout for Label Studio only
  - stable Label Studio/Postgres/public UI host
  - canonical storage host for bird and lines datasets

## Canonical Paths

### Local

- Repo root: `/Users/antoine/truenas_migration/bird_stance_classification`
- Archive root for local-only cleanup/WIP bundles: `/Users/antoine/truenas_migration/_archives/bird_stance_classification`

### `iats`

- Repo checkout: `/home/antoine/bird_stance_classification`
- Canonical training/data root: `/data/birds_project`
- Lines mirror root: `/data/lines_project`
- Repo-local `data/` is not authoritative on `iats`
- Served detector slot: `/data/birds_project/models/detector/served/model_a/current/weights.pt`
- Served attribute slot: `/data/birds_project/models/attributes/served/model_b/current/checkpoint.pt`

### TrueNAS

- Repo checkout: `/mnt/apps/code/bird_stance_classification`
- Stable UI app id: `bird-stance-classification`
- Canonical bird root: `/mnt/tank/media/birds_project`
- Canonical lines root: `/mnt/tank/media/lines_project`
- Lines import root: `/mnt/tank/media/lines_project/labelstudio/imports`
- Lines compressed mirror root: `/mnt/tank/media/lines_project/labelstudio/images_compressed/lines_bw_stilts_q60`

## Data Ownership

TrueNAS is authoritative for:

- `birds_project/raw_images`
- `birds_project/metadata`
- `birds_project/labelstudio/exports`
- `lines_project/labelstudio/imports`
- `lines_project/labelstudio/images_compressed`

`iats` owns:

- normalized datasets and derived artifacts used for training
- model training outputs
- served model slots and release metadata

## Live ML Backend Expectations

- Dockerized backend on `iats`
- GPU-backed runtime
- `MODEL_A_DEVICE=0`
- `/health` reports:
  - `status=UP`
  - `model_a_loaded=true`
  - `model_b_loaded=true`
  - `model_b_schema_version=annotation_schema_v2`
  - served weights/checkpoint paths under `/data/birds_project/models/...`

## Live Label Studio Expectations

- Public URL: [birds.ashs.live](https://birds.ashs.live)
- Stable state lives on TrueNAS only
- Project `7`: `Black Wing Stilts 1` lines batch
- Project `7` ML backend URL: `http://192.168.0.42:9090`

