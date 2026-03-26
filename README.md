# BirdSys

BirdSys is operated from three synchronized checkouts with distinct roles:

- `iats` at `/home/antoine/bird_stance_classification`: primary engineering machine, training host, experiment host, and live ML backend host.
- TrueNAS at `/mnt/apps/code/bird_stance_classification`: stable Label Studio/Postgres/public UI host for [birds.ashs.live](https://birds.ashs.live).
- Local checkout: orchestration, light tests, and repo maintenance.

The current deployed branch is `codex/isbird-schema-v2`. All three checkouts should stay on the same commit unless a rollout is actively in progress.

## Source Of Truth

- Canonical bird dataset: TrueNAS `/mnt/tank/media/birds_project`
- Canonical lines dataset: TrueNAS `/mnt/tank/media/lines_project`
- Training copy for bird models: `iats` `/home/antoine/bird_stance_classification/data/birds_project`
- Served ML backend: `iats` only
- Public Label Studio frontend: TrueNAS only

## Active Docs

- Runtime and path contract: [`ENVIRONMENT.md`](/Users/antoine/truenas_migration/bird_stance_classification/ENVIRONMENT.md)
- Deployment and verification flow: [`DEPLOYMENT.md`](/Users/antoine/truenas_migration/bird_stance_classification/DEPLOYMENT.md)
- Current live state snapshot: [`docs/current_state.md`](/Users/antoine/truenas_migration/bird_stance_classification/docs/current_state.md)
- Command-oriented workflows: [`docs/ops_workflows.md`](/Users/antoine/truenas_migration/bird_stance_classification/docs/ops_workflows.md)
- Current annotation schema comparison: [`docs/annotation_schema_current_vs_updates.md`](/Users/antoine/truenas_migration/bird_stance_classification/docs/annotation_schema_current_vs_updates.md)
- Historical docs and superseded plans: [`docs/archive/README.md`](/Users/antoine/truenas_migration/bird_stance_classification/docs/archive/README.md)

## Common Commands

Sync all checkouts to the deployed branch tip:

```bash
make iats-pull DEPLOY_BRANCH=codex/isbird-schema-v2
make truenas-pull DEPLOY_BRANCH=codex/isbird-schema-v2
```

Sync training inputs and annotation exports onto `iats`:

```bash
make iats-sync-data
make iats-import-exports DEPLOY_BRANCH=codex/isbird-schema-v2 PROJECT_ID=4 EXPORT_NAME=ann_v002_legacy
```

Train and deploy the attribute model on `iats`:

```bash
make iats-train-attributes-cv DEPLOY_BRANCH=codex/isbird-schema-v2 DATASET_DIR=/home/antoine/bird_stance_classification/data/birds_project/derived/datasets/ds_v001
make iats-train-attributes-final DEPLOY_BRANCH=codex/isbird-schema-v2 DATASET_DIR=/home/antoine/bird_stance_classification/data/birds_project/derived/datasets/ds_v001
make iats-deploy-model-b DEPLOY_BRANCH=codex/isbird-schema-v2 MODEL_B_SOURCE=/home/antoine/bird_stance_classification/data/birds_project/models/attributes/convnextv2s_v001/checkpoint.pt
```

Deploy or verify the stable frontend on TrueNAS:

```bash
make truenas-deploy-ui DEPLOY_BRANCH=codex/isbird-schema-v2
make smoke-remote
```

Prepare/import/prefill the `lines_project` batch:

```bash
make truenas-prepare-lines-batch DEPLOY_BRANCH=codex/isbird-schema-v2 LINES_BATCH_NAME=lines_bw_stilts_5k_seed_20260325_q60
make truenas-import-lines-batch DEPLOY_BRANCH=codex/isbird-schema-v2 LINES_PROJECT_ID=7 LINES_BATCH_NAME=lines_bw_stilts_5k_seed_20260325_q60
make truenas-prefill-lines-predictions DEPLOY_BRANCH=codex/isbird-schema-v2 LINES_PROJECT_ID=7 LINES_BATCH_NAME=lines_bw_stilts_5k_seed_20260325_q60 LINES_ONLY_MISSING=1
```

## Notes

- Fetch/bootstrap uses the public HTTPS origin by default so `iats` can pull without separate GitHub SSH setup.
- Live runtime paths and current project/model IDs are documented in [`docs/current_state.md`](/Users/antoine/truenas_migration/bird_stance_classification/docs/current_state.md).
