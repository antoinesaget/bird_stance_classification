# Current Live State

Last refreshed: 2026-03-26

## Repo / Branch State

- Active deployed branch: `codex/isbird-schema-v2`
- Cleanup started from synced baseline commit `a190e32`
- After this cleanup, local, `iats`, and TrueNAS should all be on the same latest commit of `codex/isbird-schema-v2`
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
- Current state:
  - `status=UP`
  - `model_a_loaded=true`
  - `model_b_loaded=true`
  - `model_b_schema_version=annotation_schema_v2`
- Detector:
  - served weights: `/home/antoine/bird_stance_classification/data/birds_project/models/detector/served/model_a/current/weights.pt`
  - current release id: `20260325T081936Z`
  - source: `/home/antoine/bird_stance_classification/yolo11m.pt`
- Attribute model:
  - served checkpoint: `/home/antoine/bird_stance_classification/data/birds_project/models/attributes/served/model_b/current/checkpoint.pt`
  - current release id: `20260325T153607Z`
  - promoted from: `/home/antoine/bird_stance_classification/data/birds_project/models/attributes/convnextv2s_v001/checkpoint.pt`
  - promotion label: `ann_v002_legacy`

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

- Stored predictions on project `7`: `4960 / 5000`
- Latest persisted prefill report summary:
  - `scanned_tasks=5000`
  - `skipped_existing=1079`
  - `predicted_tasks=3878`
  - `imported_predictions=3878`
  - `empty_predictions=43`
- Interpretation:
  - most tasks already have persisted predictions
  - remaining no-prediction tasks are concentrated in the `empty_result` subset from the report
  - this is not a frontend outage; it is a content/inference follow-up item

## Cleanup Archives

- Local archived WIP bundle:
  - `/Users/antoine/truenas_migration/_archives/bird_stance_classification/2026-03-26/local_wip`
- Local archived legacy migration snapshot:
  - `/Users/antoine/truenas_migration/_archives/bird_stance_classification/2026-03-26/full_migration_bundle`
- `iats` archived old CV artifact root:
  - `/home/antoine/bird_stance_classification/data/birds_project/models/archive/2026-03-26`
