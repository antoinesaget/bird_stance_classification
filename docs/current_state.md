# Current Live State

Last refreshed: 2026-03-26

## Repo / Branch State

- Active deployed branch: `main`
- Local, `iats`, and TrueNAS should all stay on the same latest commit of `main`
- To confirm the exact live commit on any host:

```bash
git rev-parse HEAD
```

## Host Roles

- Local
  - `/Users/antoine/truenas_migration/bird_stance_classification`
  - orchestration, tests, and repo maintenance
- `iats`
  - `/home/antoine/bird_stance_classification`
  - live ML backend
  - training and experiment host
- TrueNAS
  - `/mnt/apps/code/bird_stance_classification`
  - live Label Studio/Postgres/public UI

## Live Projects On TrueNAS

- Project `4`
  - title: `500v2`
  - tasks: `515`
  - used as the current bird annotation/export source
- Project `7`
  - title: `Black Wing Stilts 1`
  - tasks: `5000`
  - ML backend connection:
    - id: `4`
    - title: `birdsys-lines-v1`
    - url: `http://192.168.0.42:9090`
    - state: `Connected`

## Live ML Backend On `iats`

- Health endpoint: `http://127.0.0.1:9090/health`
- Durable model/data root: `/data/birds_project`
- Current state:
  - `status=UP`
  - `model_a_loaded=true`
  - `model_b_loaded=true`
  - `model_b_schema_version=annotation_schema_v2`
- Detector:
  - served weights: `/data/birds_project/models/detector/served/model_a/current/weights.pt`
  - current release id: `20260325T081936Z`
  - source: `/home/antoine/_archives/bird_stance_classification/2026-04-12/repo-root/weights/yolo11m.pt`
- Attribute model:
  - served checkpoint: `/data/birds_project/models/attributes/served/model_b/current/checkpoint.pt`
  - current release id: `20260412T140932Z`
  - promoted from: `/home/antoine/_archives/bird_stance_classification/2026-04-12/repo-root/.sandboxes/model_b_autoresearch_p7_v1/runs/20260326_150758_sampler-cap-2p5-v1/candidate/current_backend/checkpoint.pt`
  - promotion label: `archive_recovery_20260326_150758`

## `lines_project` Batch State

- Canonical batch name: `lines_bw_stilts_5k_seed_20260325_q60`
- Canonical import artifacts:
  - `/mnt/tank/media/lines_project/labelstudio/imports/lines_bw_stilts_5k_seed_20260325_q60.tasks.json`
  - `/mnt/tank/media/lines_project/labelstudio/imports/lines_bw_stilts_5k_seed_20260325_q60.manifest.csv`
  - `/mnt/tank/media/lines_project/labelstudio/imports/lines_bw_stilts_5k_seed_20260325_q60.summary.json`
  - `/mnt/tank/media/lines_project/labelstudio/imports/lines_bw_stilts_5k_seed_20260325_q60.import-report.json`
  - `/mnt/tank/media/lines_project/labelstudio/imports/lines_bw_stilts_5k_seed_20260325_q60.predictions-report.json`
- Canonical compressed mirror:
  - `/mnt/tank/media/lines_project/labelstudio/images_compressed/lines_bw_stilts_q60`
- Archived duplicate bundle:
  - `/mnt/tank/media/lines_project/labelstudio/imports/archive/2026-03-26/lines_bw_stilts_5000_seed_20260325_q60.*`

## Project 7 Prediction Prefill Status

- Stored predictions on project `7`: `5000 / 5000`
- Latest persisted prefill report summary:
  - `scanned_tasks=5000`
  - `eligible_tasks=40`
  - `skipped_existing=4960`
  - `predicted_tasks=40`
  - `imported_predictions=40`
  - `empty_predictions=40`
  - `stored_empty_predictions=40`
- Interpretation:
  - all tasks now have persisted prediction rows
  - the final 40 tasks were stored as explicit empty predictions
  - project `7` should no longer need on-the-fly backend calls for untouched tasks in the current batch

## Cleanup Archives

- Local archived WIP bundle:
  - `/Users/antoine/truenas_migration/_archives/bird_stance_classification/2026-03-26/local_wip`
- Local archived legacy migration snapshot:
  - `/Users/antoine/truenas_migration/_archives/bird_stance_classification/2026-03-26/full_migration_bundle`
- `iats` archived old CV artifact root:
  - `/home/antoine/bird_stance_classification/data/birds_project/models/archive/2026-03-26`
