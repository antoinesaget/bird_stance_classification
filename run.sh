cd /Users/antoine/bird_leg

DATA_ROOT=/Users/antoine/bird_leg/data/birds_project
TAG=$(date +%Y%m%d_%H%M%S)

INFER_DIR="$DATA_ROOT/derived/active_learning_infer/run_${TAG}_all10k_bbox"
MODEL_B_DIR="$DATA_ROOT/models/attributes/model_b_convnextv2_tiny_ds_v907_${TAG}"
PRED_DIR="$DATA_ROOT/derived/model_b_infer/run_${TAG}_resting"

# 1) Extract bbox for all images (writes predictions.parquet with detections_json)
.venv/bin/python /Users/antoine/bird_leg/scripts/infer_batch.py \
  --data-root "$DATA_ROOT" \
  --samples 0 \
  --sample-mode first \
  --imgsz 640 \
  --max-det 100 \
  --device cpu \
  --output-dir "$INFER_DIR" \
  --progress-every 50

# 2) Train Model B on latest dataset (ds_v907)
.venv/bin/python /Users/antoine/bird_leg/scripts/train_attributes.py \
  --dataset-dir "$DATA_ROOT/derived/datasets/ds_v907" \
  --config /Users/antoine/bird_leg/config/train_attributes_model_b_best.yaml \
  --data-root "$DATA_ROOT" \
  --output-dir "$MODEL_B_DIR" \
  --progress-every-batches 10

# 3) Predict with Model B on all detected birds + 4) build resting tasks.json
.venv/bin/python /Users/antoine/bird_leg/scripts/predict_resting_from_detections.py \
  --data-root "$DATA_ROOT" \
  --detections-parquet "$INFER_DIR/predictions.parquet" \
  --model-b-checkpoint "$MODEL_B_DIR/checkpoint.pt" \
  --output-dir "$PRED_DIR" \
  --tasks-json "$PRED_DIR/resting.tasks.json" \
  --behaviors resting \
  --export-crops \
  --progress-every 50

echo "tasks.json: $PRED_DIR/resting.tasks.json"
