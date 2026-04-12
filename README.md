# BirdSys

BirdSys is operated from three synchronized checkouts with distinct roles:

- `iats` at `/home/antoine/bird_stance_classification`: primary engineering machine, training host, experiment host, and live ML backend host.
- TrueNAS at `/mnt/apps/code/bird_stance_classification`: stable Label Studio/Postgres/public UI host for [birds.ashs.live](https://birds.ashs.live).
- Local checkout: orchestration, light tests, and repo maintenance.

The deployed branch is `main`. All three checkouts should stay on the same commit unless a rollout is actively in progress.

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
- Cleanup audit and reduction plan: [`docs/repo_audit_2026-04-10.md`](/Users/antoine/truenas_migration/bird_stance_classification/docs/repo_audit_2026-04-10.md)
- Historical docs and superseded plans: [`docs/archive/README.md`](/Users/antoine/truenas_migration/bird_stance_classification/docs/archive/README.md)

## Common Commands

Sync all checkouts to the deployed branch tip:

```bash
make iats-pull
make truenas-pull
```

Sync training inputs and annotation exports onto `iats`:

```bash
make iats-sync-data
make iats-import-exports PROJECT_ID=4 EXPORT_NAME=ann_v002_legacy
```

Train and deploy the attribute model on `iats`:

```bash
make iats-train-attributes-cv DATASET_DIR=/home/antoine/bird_stance_classification/data/birds_project/derived/datasets/ds_v001
make iats-train-attributes-final DATASET_DIR=/home/antoine/bird_stance_classification/data/birds_project/derived/datasets/ds_v001
make iats-deploy-model-b MODEL_B_SOURCE=/home/antoine/bird_stance_classification/data/birds_project/models/attributes/convnextv2s_v001/checkpoint.pt
```

Deploy or verify the stable frontend on TrueNAS:

```bash
make truenas-deploy-ui
make smoke-remote
```

Prepare/import/prefill the `lines_project` batch:

```bash
make truenas-prepare-lines-batch LINES_BATCH_NAME=lines_bw_stilts_5k_seed_20260325_q60
make truenas-import-lines-batch LINES_PROJECT_ID=7 LINES_BATCH_NAME=lines_bw_stilts_5k_seed_20260325_q60
make truenas-prefill-lines-predictions LINES_PROJECT_ID=7 LINES_BATCH_NAME=lines_bw_stilts_5k_seed_20260325_q60 LINES_ONLY_MISSING=1
```

Create a fresh nested-git autoresearch sandbox from the tracked template:

```bash
python3 scripts/create_autoresearch_sandbox.py --name model_b_autoresearch_p7_v2
```

## Notes

- Fetch/bootstrap uses the public HTTPS origin by default so `iats` can pull without separate GitHub SSH setup.
- Live runtime paths and current project/model IDs are documented in [`docs/current_state.md`](/Users/antoine/truenas_migration/bird_stance_classification/docs/current_state.md).
- `DEPLOY_BRANCH` defaults to `main`; only override it for temporary branch rollouts.
