# End-to-End Runbook

## 1) Setup
```bash
cd /Users/antoine/bird_leg
cp .env.example .env
make bootstrap
test -f "${MODEL_A_WEIGHTS}"
```

## 2) Initialize data root and metadata
```bash
# One-time copy into canonical raw_images (container-safe; no external symlink dependency)
mkdir -p "${BIRDS_DATA_ROOT}/raw_images/scolop2"
rsync -a --ignore-existing /Users/antoine/bird_leg/scraped_images/scolop2_10k/ "${BIRDS_DATA_ROOT}/raw_images/scolop2/"

uv run python /Users/antoine/bird_leg/scripts/register_images.py \
  --source-dir "${BIRDS_DATA_ROOT}/raw_images/scolop2" \
  --site-id scolop2 \
  --data-root "${BIRDS_DATA_ROOT}" \
  --link-mode none \
  --overwrite-metadata
```

## 3) Prepare a 50-image Label Studio sample import bundle
```bash
uv run python /Users/antoine/bird_leg/scripts/prepare_labelstudio_import.py \
  --data-root "${BIRDS_DATA_ROOT}" \
  --site-id scolop2 \
  --count 50 \
  --sample-mode first
```

## 4) Start services
```bash
docker compose --env-file /Users/antoine/bird_leg/.env -f /Users/antoine/bird_leg/deploy/docker-compose.yml up -d
```

## 5) Annotate and export
- Create project in Label Studio using `/Users/antoine/bird_leg/labelstudio/label_config.xml`.
- In project settings:
  - disable `Interactive preannotations`
  - keep `Auto-Accept Suggestions` disabled during manual QA
- Import `/Users/antoine/bird_leg/data/birds_project/labelstudio/imports/scolop2_sample50.tasks.json` for the sample batch, or import from `${BIRDS_DATA_ROOT}/raw_images/scolop2`.
- Export annotations as `ann_vXXX.json` into `${BIRDS_DATA_ROOT}/labelstudio/exports`.

## 6) Normalize + crops + dataset
```bash
uv run python /Users/antoine/bird_leg/scripts/export_normalize.py \
  --export-json "${BIRDS_DATA_ROOT}/labelstudio/exports/ann_v001.json" \
  --annotation-version ann_v001 \
  --data-root "${BIRDS_DATA_ROOT}"

uv run python /Users/antoine/bird_leg/scripts/make_crops.py \
  --annotation-version ann_v001 \
  --data-root "${BIRDS_DATA_ROOT}"

uv run python /Users/antoine/bird_leg/scripts/build_dataset.py \
  --annotation-version ann_v001 \
  --data-root "${BIRDS_DATA_ROOT}"
```

## 7) Train Model B and C
```bash
uv run python /Users/antoine/bird_leg/scripts/train_attributes.py \
  --dataset-dir "${BIRDS_DATA_ROOT}/derived/datasets/ds_v001" \
  --data-root "${BIRDS_DATA_ROOT}"

uv run python /Users/antoine/bird_leg/scripts/train_image_status.py \
  --annotation-version ann_v001 \
  --data-root "${BIRDS_DATA_ROOT}"
```

## 8) Active learning inference and batch selection
```bash
uv run python /Users/antoine/bird_leg/scripts/infer_batch.py \
  --data-root "${BIRDS_DATA_ROOT}" \
  --samples 2000

uv run python /Users/antoine/bird_leg/scripts/select_active_learning_batch.py \
  --predictions "${BIRDS_DATA_ROOT}/derived/active_learning_infer/run_YYYYMMDD_HHMMSS/predictions.parquet" \
  --batch-size 300
```

## 9) Backend latency smoke benchmark (20 images)
```bash
docker compose --env-file /Users/antoine/bird_leg/.env -f /Users/antoine/bird_leg/deploy/docker-compose.yml \
  exec -T ml-backend python -m services.ml_backend.app.benchmark_predict \
  --images-dir /data/birds_project/raw_images/scolop2 \
  --samples 20 \
  --sample-mode first
```
