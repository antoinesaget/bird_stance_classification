# Ops Workflows

## 1) Keep The Three Checkouts In Sync

Use the same branch on local, `iats`, and TrueNAS. The current live branch is `codex/isbird-schema-v2`.

```bash
make iats-pull DEPLOY_BRANCH=codex/isbird-schema-v2
make truenas-pull DEPLOY_BRANCH=codex/isbird-schema-v2
```

## 2) Bird Annotation Export And Training Flow

Export annotations from TrueNAS:

```bash
make truenas-export-annotations DEPLOY_BRANCH=codex/isbird-schema-v2 PROJECT_ID=4 EXPORT_NAME=ann_v002_legacy
```

Copy exports to `iats`:

```bash
make iats-import-exports DEPLOY_BRANCH=codex/isbird-schema-v2 PROJECT_ID=4 EXPORT_NAME=ann_v002_legacy
```

Or sync all canonical bird inputs:

```bash
make iats-sync-data
```

Run attribute-model CV:

```bash
make iats-train-attributes-cv DEPLOY_BRANCH=codex/isbird-schema-v2 DATASET_DIR=/home/antoine/bird_stance_classification/data/birds_project/derived/datasets/ds_v001
```

Run final attribute training:

```bash
make iats-train-attributes-final DEPLOY_BRANCH=codex/isbird-schema-v2 DATASET_DIR=/home/antoine/bird_stance_classification/data/birds_project/derived/datasets/ds_v001
```

Deploy the trained Model B checkpoint:

```bash
make iats-deploy-model-b DEPLOY_BRANCH=codex/isbird-schema-v2 MODEL_B_SOURCE=/home/antoine/bird_stance_classification/data/birds_project/models/attributes/convnextv2s_v001/checkpoint.pt PROMOTION_LABEL=ann_v002_legacy
```

## 3) TrueNAS Frontend Deploy Flow

```bash
make truenas-deploy-ui DEPLOY_BRANCH=codex/isbird-schema-v2
make smoke-remote
```

Use this after repo changes that affect the stable Label Studio frontend, local-files override wiring, or app config.

## 4) `lines_project` Batch Flow

Prepare the `q60` mirror and task bundle:

```bash
make truenas-prepare-lines-batch DEPLOY_BRANCH=codex/isbird-schema-v2 \
  LINES_PROJECT_ID=7 \
  LINES_BATCH_NAME=lines_bw_stilts_5k_seed_20260325_q60 \
  LINES_JPEG_QUALITY=60 \
  LINES_SAMPLE_SIZE=5000 \
  LINES_SAMPLE_SEED=20260325
```

Import the task bundle:

```bash
make truenas-import-lines-batch DEPLOY_BRANCH=codex/isbird-schema-v2 \
  LINES_PROJECT_ID=7 \
  LINES_BATCH_NAME=lines_bw_stilts_5k_seed_20260325_q60
```

Persist predictions instead of generating them on the fly:

```bash
make truenas-prefill-lines-predictions DEPLOY_BRANCH=codex/isbird-schema-v2 \
  LINES_PROJECT_ID=7 \
  LINES_BATCH_NAME=lines_bw_stilts_5k_seed_20260325_q60 \
  LINES_ONLY_MISSING=1
```

## 5) What To Check When Something Looks Wrong

- Repo drift:

```bash
git status --short
ssh iats 'cd /home/antoine/bird_stance_classification && git status --short'
ssh truenas 'cd /mnt/apps/code/bird_stance_classification && git status --short'
```

- ML backend health:

```bash
ssh iats 'curl -s http://127.0.0.1:9090/health | jq .'
```

- TrueNAS app health:

```bash
ssh truenas 'midclt call app.get_instance bird-stance-classification'
```

- Project `7` prediction coverage:

```bash
ssh truenas 'cd /mnt/apps/code/bird_stance_classification && python3 - <<'"'"'\"'\"'PY'\"'\"'\nfrom pathlib import Path\nfrom scripts.export_labelstudio_snapshot import resolve_api_token, request_json\nbase_url=\"http://127.0.0.1:30280\"\napi_token=\"\"\nfor line in Path(\"deploy/env/truenas.env\").read_text().splitlines():\n    if line.startswith(\"LABEL_STUDIO_API_TOKEN=\"):\n        api_token=line.split(\"=\",1)[1].strip(); break\nresolved=resolve_api_token(base_url, api_token)\nprint(request_json(base_url, \"/api/tasks?project=7&page_size=1\", resolved)[\"total_predictions\"])\nPY'
```
