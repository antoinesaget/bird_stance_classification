# BirdSys

BirdSys is a monorepo with four operational subprojects and one internal shared core:

```text
projects/
  labelstudio/
  datasets/
  ml_backend/
  ml_experiments/
shared/
  birdsys_core/
```

The deployed branch is `main`. Local and `iats` should be identical full checkouts on the same commit. TrueNAS should stay on the same commit too, but only keeps the minimal deployment checkout for Label Studio.

## Host Roles

- Local: full engineering checkout, orchestration, tests, repo maintenance
- `iats` at `/home/antoine/bird_stance_classification`: full engineering checkout identical to local, training host, experiment host, live ML backend host
- TrueNAS at `/mnt/apps/code/bird_stance_classification`: minimal Label Studio deployment checkout, stable Label Studio/Postgres/public UI host for [birds.ashs.live](https://birds.ashs.live)

## Subprojects

- `[projects/labelstudio/README.md](/Users/antoine/truenas_migration/bird_stance_classification/projects/labelstudio/README.md)`: Label Studio exports, task import, compressed image batches, prediction refresh, TrueNAS app files
- `[projects/datasets/README.md](/Users/antoine/truenas_migration/bird_stance_classification/projects/datasets/README.md)`: annotation normalization, crop generation, dataset versioning and reports
- `[projects/ml_backend/README.md](/Users/antoine/truenas_migration/bird_stance_classification/projects/ml_backend/README.md)`: served backend runtime, promotion, `iats` deploy assets
- `[projects/ml_experiments/README.md](/Users/antoine/truenas_migration/bird_stance_classification/projects/ml_experiments/README.md)`: training, CV, evaluation, autoresearch sandboxes
- `[shared/birdsys_core/README.md](/Users/antoine/truenas_migration/bird_stance_classification/shared/birdsys_core/README.md)`: shared contracts, config, taxonomy, artifact schema

## Active Docs

- Runtime and path contract: `[ENVIRONMENT.md](/Users/antoine/truenas_migration/bird_stance_classification/ENVIRONMENT.md)`
- Deployment and verification flow: `[DEPLOYMENT.md](/Users/antoine/truenas_migration/bird_stance_classification/DEPLOYMENT.md)`
- Live state snapshot: `[docs/current_state.md](/Users/antoine/truenas_migration/bird_stance_classification/docs/current_state.md)]` 

## Notes

- TrueNAS is authoritative for canonical bird and lines storage.
- `iats` owns derived training artifacts and the served ML backend.

