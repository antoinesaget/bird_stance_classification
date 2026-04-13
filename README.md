# BirdSys

BirdSys is now a monorepo with four operational subprojects and one internal shared core:

```text
projects/
  labelstudio/
  datasets/
  ml_backend/
  ml_experiments/
shared/
  birdsys_core/
ops/
docs/
archived/
```

The deployed branch is `main`. Local, `iats`, and TrueNAS should stay on the same commit unless a rollout is in progress.

## Host Roles

- Local: orchestration, tests, repo maintenance
- `iats` at `/home/antoine/bird_stance_classification`: training host, experiment host, live ML backend host
- TrueNAS at `/mnt/apps/code/bird_stance_classification`: stable Label Studio/Postgres/public UI host for [birds.ashs.live](https://birds.ashs.live)

## Subprojects

- [`projects/labelstudio/README.md`](/Users/antoine/truenas_migration/bird_stance_classification/projects/labelstudio/README.md): Label Studio exports, task import, compressed image batches, prediction refresh, TrueNAS app files
- [`projects/datasets/README.md`](/Users/antoine/truenas_migration/bird_stance_classification/projects/datasets/README.md): annotation normalization, crop generation, dataset versioning and reports
- [`projects/ml_backend/README.md`](/Users/antoine/truenas_migration/bird_stance_classification/projects/ml_backend/README.md): served backend runtime, promotion, `iats` deploy assets
- [`projects/ml_experiments/README.md`](/Users/antoine/truenas_migration/bird_stance_classification/projects/ml_experiments/README.md): training, CV, evaluation, autoresearch sandboxes
- [`shared/birdsys_core/README.md`](/Users/antoine/truenas_migration/bird_stance_classification/shared/birdsys_core/README.md): shared contracts, config, taxonomy, artifact schema

## Umbrella CLI

The public Python surface is the workspace CLI:

```bash
birdsys labelstudio ...
birdsys datasets ...
birdsys backend ...
birdsys experiment ...
```

Examples:

```bash
birdsys labelstudio export-snapshot --help
birdsys datasets build-dataset --help
birdsys backend promote-model --help
birdsys experiment create-autoresearch-sandbox --help
```

## Make Targets

The root `Makefile` is now only a thin dispatcher for local compose actions and remote workflows:

```bash
make iats-pull
make truenas-pull
make iats-sync-data
make iats-train-attributes-cv DATASET_DIR=/data/birds_project/derived/datasets/ds_v001
make truenas-prepare-lines-batch LINES_BATCH_NAME=lines_bw_stilts_5k_seed_20260325_q60
make smoke-remote
```

Create a fresh autoresearch sandbox from the tracked template:

```bash
birdsys experiment create-autoresearch-sandbox --name model_b_autoresearch_p7_v2
```

## Active Docs

- Runtime and path contract: [`ENVIRONMENT.md`](/Users/antoine/truenas_migration/bird_stance_classification/ENVIRONMENT.md)
- Deployment and verification flow: [`DEPLOYMENT.md`](/Users/antoine/truenas_migration/bird_stance_classification/DEPLOYMENT.md)
- Live state snapshot: [`docs/current_state.md`](/Users/antoine/truenas_migration/bird_stance_classification/docs/current_state.md)
- Command workflows: [`docs/ops_workflows.md`](/Users/antoine/truenas_migration/bird_stance_classification/docs/ops_workflows.md)
- Annotation schema notes: [`docs/annotation_schema_current_vs_updates.md`](/Users/antoine/truenas_migration/bird_stance_classification/docs/annotation_schema_current_vs_updates.md)
- Historical docs and superseded plans: [`docs/archive/README.md`](/Users/antoine/truenas_migration/bird_stance_classification/docs/archive/README.md)

## Notes

- TrueNAS is authoritative for canonical bird and lines storage.
- `iats` owns derived training artifacts and the served ML backend.
- Anything not clearly on the annotation -> dataset -> experiment -> promotion -> Label Studio loop belongs under `archived/`.
