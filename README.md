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

## Notes

- TrueNAS is authoritative for canonical bird and lines storage.
- `iats` owns derived training artifacts and the served ML backend.

