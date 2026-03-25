# BirdSys

BirdSys is now organized around one primary engineering host and one stable production UI host:

- `iats` (`/home/antoine/bird_stance_classification`): main dev box, training box, experiment box, and production ML backend host.
- TrueNAS (`/mnt/apps/code/bird_stance_classification` + `bird-stance-classification` app): stable Label Studio/Postgres/public UI at [birds.ashs.live](https://birds.ashs.live).
- Local: low-priority orchestration, light smoke tests, and repo maintenance.

## Source Of Truth

- Canonical annotation/data root: TrueNAS `/mnt/tank/media/birds_project`
- Training copy: `iats` `/home/antoine/bird_stance_classification/data/birds_project`
- Stable public frontend: TrueNAS only
- Stable ML backend: `iats` only

## Managed Runtime Files

- Local debug stack: `deploy/docker-compose.local.yml`
- `iats` ML stack: `deploy/docker-compose.iats-ml.yml`
- TrueNAS UI stack: `deploy/docker-compose.truenas.yml`
- Environment templates: `deploy/env/*.example`
- Operations scripts: `scripts/ops/*.sh`

Legacy files `deploy/docker-compose.yml` and `deploy/docker-compose.app-only.yml` are retained only for reference. New work should use the explicit target files above.

## Common Commands

Prepare local tooling:

```bash
uv sync --python 3.11
cp deploy/env/local.env.example deploy/env/local.env
make local-config
```

Pull or bootstrap remote checkouts:

```bash
make iats-pull
make truenas-pull
```

Export annotations from TrueNAS and copy them to `iats`:

```bash
make iats-import-exports PROJECT_ID=4 ANNOTATION_VERSION=ann_v001
```

Sync canonical training inputs from TrueNAS to `iats`:

```bash
make iats-sync-data
```

Run training on `iats`:

```bash
make iats-train TRAIN_PIPELINE=attributes DATASET_VERSION=ds_v001
```

Promote a candidate model and restart the served ML backend on `iats`:

```bash
make iats-promote-model PROMOTION_SOURCE=data/birds_project/models/detector/experiments/run_001/weights.pt
make iats-deploy-ml
```

Deploy the stable frontend on TrueNAS:

```bash
make truenas-deploy-ui
```

Cross-host smoke checks:

```bash
make smoke-remote
```

## Git Notes

- The operational model assumes `main` is the deploy branch.
- The current repo may still be on a feature branch during migration; reconcile that branch into `main` before using the remote deploy flow as the long-term default.
- `iats` currently cannot fetch from `git@github.com:...` without separate GitHub SSH credentials. The managed pull flow therefore uses the public HTTPS repository URL for fetch/bootstrap. Push from `iats` still requires explicit GitHub credentials on that machine.

## More Detail

- Deployment/runbook: `DEPLOYMENT.md`
- Environment contract and artifact layout: `ENVIRONMENT.md`
