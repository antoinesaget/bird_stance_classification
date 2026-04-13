# Ops Workflows

## 1) Keep The Three Checkouts In Sync

Use the same branch on local, `iats`, and TrueNAS. The live branch is `main`.

```bash
make iats-pull
make truenas-pull
```

## 2) Bird Annotation Export And Training Flow

Export annotations from TrueNAS:

```bash
make truenas-export-annotations PROJECT_ID=4 EXPORT_NAME=ann_v002_legacy
```

Copy exports to `iats`:

```bash
make iats-import-exports PROJECT_ID=4 EXPORT_NAME=ann_v002_legacy
```

Or sync all canonical bird inputs:

```bash
make iats-sync-data
```

Run attribute-model CV:

```bash
make iats-train-attributes-cv DATASET_DIR=/data/birds_project/derived/datasets/ds_v001
```

Run final attribute training:

```bash
make iats-train-attributes-final DATASET_DIR=/data/birds_project/derived/datasets/ds_v001
```

Deploy the trained Model B checkpoint:

```bash
make iats-deploy-model-b MODEL_B_SOURCE=/data/birds_project/models/attributes/convnextv2s_v001/checkpoint.pt PROMOTION_LABEL=ann_v002_legacy
```

## 3) TrueNAS Frontend Deploy Flow

```bash
make truenas-deploy-ui
make smoke-remote
```

Use this after repo changes that affect the stable Label Studio frontend, local-files override wiring, or app config.

## 4) `lines_project` Batch Flow

Prepare the `q60` mirror and task bundle:

```bash
make truenas-prepare-lines-batch \
  LINES_PROJECT_ID=7 \
  LINES_BATCH_NAME=lines_bw_stilts_5k_seed_20260325_q60 \
  LINES_JPEG_QUALITY=60 \
  LINES_SAMPLE_SIZE=5000 \
  LINES_SAMPLE_SEED=20260325
```

Import the task bundle:

```bash
make truenas-import-lines-batch \
  LINES_PROJECT_ID=7 \
  LINES_BATCH_NAME=lines_bw_stilts_5k_seed_20260325_q60
```

Persist predictions instead of generating them on the fly:

```bash
make truenas-prefill-lines-predictions \
  LINES_PROJECT_ID=7 \
  LINES_BATCH_NAME=lines_bw_stilts_5k_seed_20260325_q60 \
  LINES_ONLY_MISSING=1
```

Refresh untouched tasks with the current ML backend, pilot first:

```bash
make truenas-refresh-lines-predictions \
  LINES_PROJECT_ID=7 \
  LINES_BATCH_NAME=lines_bw_stilts_5k_seed_20260325_q60 \
  LINES_LIMIT=100
```

Run the full untouched-task refresh after the pilot:

```bash
make truenas-refresh-lines-predictions \
  LINES_PROJECT_ID=7 \
  LINES_BATCH_NAME=lines_bw_stilts_5k_seed_20260325_q60 \
  LINES_LIMIT=0
```

Project-7-specific specialist refresh and reporting flows were retired from the active ops surface on April 12, 2026. Their code was moved under `archived/legacy/` or host-specific dated archive roots.

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
ssh truenas 'cd /mnt/apps/code/bird_stance_classification && PYTHONPATH="ops/src:shared/birdsys_core/src:projects/labelstudio/src" python3 - <<'"'"'\"'\"'PY'\"'\"'\nfrom pathlib import Path\nfrom birdsys.labelstudio.export_snapshot import resolve_api_token, request_json\nbase_url = \"http://127.0.0.1:30280\"\napi_token = \"\"\nfor line in Path(\"projects/labelstudio/deploy/env/truenas.env\").read_text().splitlines():\n    if line.startswith(\"LABEL_STUDIO_API_TOKEN=\"):\n        api_token = line.split(\"=\", 1)[1].strip()\n        break\nresolved = resolve_api_token(base_url, api_token)\nprint(request_json(base_url, \"/api/tasks?project=7&page_size=1\", resolved)[\"total_predictions\"])\nPY'
```

- Project `7` annotation and untouched-task counts:

```bash
ssh truenas 'cd /mnt/apps/code/bird_stance_classification && PYTHONPATH="ops/src:shared/birdsys_core/src:projects/labelstudio/src" python3 - <<'"'"'\"'\"'PY'\"'\"'\nfrom pathlib import Path\nfrom birdsys.labelstudio.export_snapshot import resolve_api_token, request_json\nbase_url = \"http://127.0.0.1:30280\"\napi_token = \"\"\nfor line in Path(\"projects/labelstudio/deploy/env/truenas.env\").read_text().splitlines():\n    if line.startswith(\"LABEL_STUDIO_API_TOKEN=\"):\n        api_token = line.split(\"=\", 1)[1].strip()\n        break\nresolved = resolve_api_token(base_url, api_token)\npage = 1\npage_size = 200\nannotated = 0\nuntouched = 0\nwhile True:\n    payload = request_json(base_url, f\"/api/tasks?project=7&page={page}&page_size={page_size}\", resolved)\n    tasks = payload.get(\"tasks\") or []\n    if not tasks:\n        break\n    for task in tasks:\n        drafts = len(task.get(\"drafts\") or [])\n        if int(task.get(\"total_annotations\") or 0) > 0 or bool(task.get(\"is_labeled\")):\n            annotated += 1\n        elif drafts == 0:\n            untouched += 1\n    total = int(payload.get(\"total\") or 0)\n    if page * page_size >= total:\n        break\n    page += 1\nprint({\"annotated\": annotated, \"untouched\": untouched})\nPY'
```

- Sample refreshed tasks still have exactly one prediction row and a newer prediction timestamp:

```bash
ssh truenas 'cd /mnt/apps/code/bird_stance_classification && PYTHONPATH="ops/src:shared/birdsys_core/src:projects/labelstudio/src" python3 - <<'"'"'\"'\"'PY'\"'\"'\nfrom pathlib import Path\nfrom birdsys.labelstudio.export_snapshot import resolve_api_token, request_json\nbase_url = \"http://127.0.0.1:30280\"\napi_token = \"\"\nrun_started_at = \"2026-03-31T00:00:00Z\"\nfor line in Path(\"projects/labelstudio/deploy/env/truenas.env\").read_text().splitlines():\n    if line.startswith(\"LABEL_STUDIO_API_TOKEN=\"):\n        api_token = line.split(\"=\", 1)[1].strip()\n        break\nresolved = resolve_api_token(base_url, api_token)\npayload = request_json(base_url, \"/api/tasks?project=7&page=1&page_size=100\", resolved)\nseen = 0\nfor task in payload.get(\"tasks\") or []:\n    if int(task.get(\"total_annotations\") or 0) > 0:\n        continue\n    detail = request_json(base_url, f\"/api/tasks/{task['id']}\", resolved)\n    predictions = detail.get(\"predictions\") or []\n    if not predictions:\n        continue\n    latest = max(str(pred.get(\"updated_at\") or \"\") for pred in predictions)\n    print({\"task\": task[\"id\"], \"total_predictions\": detail.get(\"total_predictions\"), \"latest_prediction_updated_at\": latest, \"newer_than_run_start\": latest >= run_started_at})\n    seen += 1\n    if seen >= 5:\n        break\nPY'
```
