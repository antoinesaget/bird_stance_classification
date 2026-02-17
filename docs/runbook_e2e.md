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
uv run python /Users/antoine/bird_leg/scripts/register_images.py \
  --source-dir /Users/antoine/bird_leg/scraped_images/scolop2_10k \
  --site-id scolop2 \
  --data-root "${BIRDS_DATA_ROOT}" \
  --link-mode symlink
```

## 3) Start services
```bash
docker compose --env-file /Users/antoine/bird_leg/.env -f /Users/antoine/bird_leg/deploy/docker-compose.yml up -d
```

## 4) Annotate and export
- Create project in Label Studio using `/Users/antoine/bird_leg/labelstudio/label_config.xml`.
- Import images from `${BIRDS_DATA_ROOT}/raw_images/scolop2`.
- Export annotations as `ann_vXXX.json` into `${BIRDS_DATA_ROOT}/labelstudio/exports`.

## 5) Normalize + crops + dataset
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

## 6) Train Model B and C
```bash
uv run python /Users/antoine/bird_leg/scripts/train_attributes.py \
  --dataset-dir "${BIRDS_DATA_ROOT}/derived/datasets/ds_v001" \
  --data-root "${BIRDS_DATA_ROOT}"

uv run python /Users/antoine/bird_leg/scripts/train_image_status.py \
  --annotation-version ann_v001 \
  --data-root "${BIRDS_DATA_ROOT}"
```

## 7) Active learning inference and batch selection
```bash
uv run python /Users/antoine/bird_leg/scripts/infer_batch.py \
  --data-root "${BIRDS_DATA_ROOT}" \
  --samples 2000

uv run python /Users/antoine/bird_leg/scripts/select_active_learning_batch.py \
  --predictions "${BIRDS_DATA_ROOT}/derived/active_learning_infer/run_YYYYMMDD_HHMMSS/predictions.parquet" \
  --batch-size 300
```
