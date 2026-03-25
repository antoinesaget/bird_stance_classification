# BirdSys Environment Contract

## Runtime Roles

- Local: optional debug stack and orchestration
- `iats`: dev, training, experiment tracking, and production ML serving
- TrueNAS: stable Label Studio/Postgres/public UI and canonical storage

## Canonical Paths

### TrueNAS

- Repo checkout: `/mnt/apps/code/bird_stance_classification`
- Stable UI app id: `bird-stance-classification`
- Canonical data root: `/mnt/tank/media/birds_project`

### `iats`

- Repo checkout: `/home/antoine/bird_stance_classification`
- Training copy root: `/home/antoine/bird_stance_classification/data/birds_project`
- Served detector slot: `/home/antoine/bird_stance_classification/data/birds_project/models/detector/served/model_a/current/weights.pt`

### Local

- Repo root: current workspace
- Optional local data root: `.local/birds_project`

## Tracked Env Templates

- `deploy/env/local.env.example`
- `deploy/env/iats.env.example`
- `deploy/env/truenas.env.example`

The managed scripts expect real untracked env files with the same names minus `.example`.

## Canonical Data Ownership

TrueNAS is the source of truth for:

- `raw_images/`
- `metadata/`
- `labelstudio/exports/`

`iats` keeps a synced working copy of those inputs for training. Derived artifacts and experiments remain local to `iats`.

## Artifact Layout

Under `${BIRDS_DATA_ROOT}`:

- `raw_images/`
- `metadata/`
- `labelstudio/exports/`
- `labelstudio/normalized/`
- `derived/crops/`
- `derived/datasets/`
- `models/detector/`
- `models/attributes/`
- `models/image_status/`

Detector serving layout on `iats`:

- `models/detector/served/model_a/current/weights.pt`
- `models/detector/served/model_a/current/promotion.json`
- `models/detector/served/model_a/releases/<timestamp>/weights.pt`
- `models/detector/served/model_a/releases/<timestamp>/promotion.json`

## ML Backend Expectations

- Dockerized backend on `iats`
- GPU-backed runtime (`gpus: all`)
- `MODEL_A_DEVICE=0`
- `/health` should report:
  - `model_a_loaded=true`
  - a non-CPU `model_a_device`
  - the served weights path
  - promotion metadata when `promotion.json` exists

## Label Studio Expectations

- Public URL: `https://birds.ashs.live`
- Stable state lives on TrueNAS only
- TrueNAS app mounts:
  - canonical `birds_project`
  - optional `lines_project`
  - persistent Postgres data dir
  - persistent Label Studio app data dir
  - repo-tracked `deploy/overrides/localfiles_views.py`

## Verification Commands

Smoke tests:

```bash
uv run pytest -q tests/smoke
```

Local compose render:

```bash
ENV_FILE=deploy/env/local.env.example scripts/ops/local_compose.sh config
```

`iats` backend health:

```bash
ssh iats 'curl -sS http://127.0.0.1:9090/health'
```

TrueNAS app state:

```bash
ssh truenas 'midclt call app.get_instance bird-stance-classification'
```
