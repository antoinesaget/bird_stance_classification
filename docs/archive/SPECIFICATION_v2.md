# Bird Annotation and Classification System - Specification v2 (No Label Studio)

## 0) Objective
Build a fully self-hosted system to annotate bird images, train models, and run active learning loops.

This v2 specification replaces the prior Label Studio based workflow with a custom annotation platform built from scratch.

The system must support:
- 0..N birds per image.
- Image-level usability labeling.
- Bird-level box + nested attribute labeling with strict conditional logic.
- ML-assisted pre-annotations.
- Deterministic export normalization and dataset/version lineage.
- Model training and inference pipelines for active learning.

## 1) Hard constraints
- No Label Studio dependency.
- No MinIO/S3 dependency.
- No SAM refinement.
- Detector (Model A) uses pretrained YOLO weights; no detector fine-tuning in this phase.
- Active learning is manual and driven by attribute model uncertainty.
- Must run on:
  - Linux + NVIDIA RTX 3090 (CUDA path, primary heavy training/inference)
  - macOS Apple Silicon (M4) (MPS/CPU path, dev and light inference)
- Max annotators concurrently: 3.
- Python baseline: 3.11.x.

## 2) System architecture
Implement these first-class components:

1. Raw data and metadata layer
- Immutable raw image store on local disk.
- Metadata registry parquet with one row per image.

2. Custom annotation web app (new)
- Two-pane image annotation UI.
- Per-image and per-region labeling with strict nested conditions.
- Save annotation revisions and task states.
- Consume ML prediction suggestions and allow edit/accept.

3. ML inference backend service
- Model A: YOLO bird detection boxes.
- Model B: bird attribute predictions (readability/specie/behavior/substrate/legs).
- Model C: image_status prediction.
- Single call that returns full prediction bundle for one or many tasks.

4. Deterministic normalization pipeline
- Convert annotation exports to canonical parquet tables.
- Apply masking/default logic deterministically.

5. Crop generation pipeline
- Generate bird crops from normalized boxes with configurable margin.

6. Dataset builder
- Build train/val/test parquet splits deterministically.
- Emit dataset manifest with lineage and label counts.

7. Training pipelines
- Model B multi-head training with masked heads.
- Model C binary image-level training.

8. Active learning inference + selection
- Run pooled inference.
- Score/select batches with uncertainty + diversity + random exploration.

## 3) Canonical taxonomy (authoritative)

### 3.1 Image-level label (required)
`image_status`:
- `has_usable_birds`
- `no_usable_birds`

### 3.2 Bird region label
- Rectangle label type: `Bird`

### 3.3 Bird attributes
Required for every bird region:
- `readability`:
  - `readable`
  - `occluded`
  - `unreadable`
- `specie`:
  - `correct`
  - `incorrect`
  - `unsure`

Conditional fields:
- Show/require `behavior` and `substrate` only if:
  - `readability != unreadable`
  - AND `specie != incorrect`

`behavior` classes:
- `resting`
- `foraging`
- `flying`
- `backresting`
- `preening`
- `display`
- `unsure`

`substrate` classes:
- `ground`
- `water`
- `air`
- `unsure`

- Show/require `legs` only if:
  - `behavior in {resting, backresting}`
  - AND `substrate in {ground, water}`

`legs` classes:
- `one`
- `two`
- `unsure`
- `sitting`

### 3.4 Terminal logic
For each bird region:
- If `readability = unreadable`, region classification ends.
- Else if `specie = incorrect`, region classification ends.
- Else classify `behavior` + `substrate`.
- Add `legs` only in resting/backresting on ground/water subspace.

## 4) Annotation UI requirements
Implement a custom annotation frontend with these behaviors:

1. Layout and interaction
- Two-column layout:
  - Left: image + bboxes.
  - Right: image-level and region-level controls.
- Zoom/pan image.
- Draw/edit/delete Bird rectangles.
- Region selection drives right panel context.

2. Validation
- Block submit if image_status missing.
- Block submit if required fields for visible branch are missing.
- Never require hidden conditional fields.

3. Suggested hotkeys (match established workflow)
- `image_status`: `a`=has_usable_birds, `q`=no_usable_birds
- `readability`: `z`=readable, `s`=occluded, `x`=unreadable
- `specie`: `e`=correct, `d`=incorrect, `c`=unsure
- `behavior`: `r`=resting, `f`=foraging, `v`=flying
- `substrate`: `t`=ground, `g`=water, `b`=air
- `legs`: `y`=one, `h`=two, `n`=unsure

4. Prediction integration UX
- For each task, show model suggestions for boxes/attributes/image_status.
- Annotator can accept, edit, or ignore suggestions.
- Manual edits must persist after save/update.
- Label-all and single-task update flows must behave consistently.

5. Throughput requirement
- Designed for fast keyboard-first operation with minimal scrolling.

## 5) Annotation API and persistence contract
Create a first-party annotation backend (replace Label Studio APIs).

Minimum required entities:
- `Project`
- `Task` (image assignment unit)
- `Annotation` (human-labeled state)
- `Prediction` (model-suggested state)
- `Region` (bbox + per-region labels)

### 5.1 Task payload shape (minimum)
Each task must contain:
- `task_id`
- `image_id`
- `image_path` (or resolvable image URL)
- optional metadata (`site_id`, timestamp, split tags)

### 5.2 Stored annotation shape (minimum)
For each annotation version:
- image-level:
  - `image_status`
- region-level list:
  - `bird_id`
  - bbox normalized (`x`, `y`, `w`, `h`) in [0,1]
  - `readability`, `specie`, optional `behavior`, `substrate`, `legs`

### 5.3 Prediction serving API (required)
`POST /predict`
- Input: list of tasks with local image path/URL.
- Output: list of predictions; each prediction includes:
  - task id
  - overall score (image_status confidence)
  - model version
  - list of result items:
    - bbox items
    - per-region choice items
    - one image_status choice item

Also expose health and setup endpoints for tooling compatibility:
- `GET /health`
- `GET /healthz`
- `POST /setup`
- `POST /validate`

## 6) ML prediction rules (runtime)

### 6.1 Model A (detector)
- YOLO pretrained weights.
- Output normalized bboxes + confidence + label `Bird`.

### 6.2 Model B (attributes)
Outputs per detection:
- readability + confidence
- specie + confidence
- behavior + confidence
- substrate + confidence
- legs + confidence

Inference serialization rule:
- Always emit `readability` and `specie` suggestions.
- Emit `behavior` and `substrate` only when `readability != unreadable` and `specie != incorrect`.
- Emit `legs` only when behavior in `{resting, backresting}` and substrate in `{ground, water}`.

### 6.3 Model C (image status)
- Predict `has_usable_birds` probability.
- Return final label and confidence as image-level suggestion.

### 6.4 Fallback behavior
If trained checkpoints for B/C are unavailable, backend must still run using deterministic heuristic predictors.

## 7) Canonical storage layout
Use a deterministic versioned disk layout:

```text
<data_root>/
  raw_images/                         # immutable
    <site_id>/...
  metadata/
    images.parquet                    # one row per image

  annotations/
    exports/                          # immutable raw annotation exports (json)
      ann_v001.json
      ann_v002.json
    normalized/
      ann_v001/
        images_labels.parquet
        birds.parquet
      ann_v002/
        ...

  derived/
    crops/
      ann_v001/
        <bird_crop>.jpg
        _crops.parquet
    datasets/
      ds_v001/
        train.parquet
        val.parquet
        test.parquet
        manifest.json

    active_learning_infer/
      run_YYYYMMDD_HHMMSS/
        predictions.parquet
        run_config.json
        selection_YYYYMMDD_HHMMSS/
          selected_images.csv
          selection_report.json

    benchmarks/
      ml_backend/
        predict_latency_YYYYMMDD_HHMMSS/
          latency_samples.csv
          latency_report.json

  models/
    detector/
    attributes/
      convnextv2s_vXXX/
        config.yaml
        checkpoint.pt
        metrics.json
        legs_confusion_matrix.csv
    image_status/
      status_vXXX/
        config.yaml
        checkpoint.pt
        metrics.json
```

Rules:
- Never mutate `raw_images`.
- Never overwrite previous `ann_vXXX`, `ds_vXXX`, or model version folders.
- All version outputs must be append-only.

## 8) Metadata indexing
Implement image registry script behavior:
- Scan source images (`jpg/jpeg/png/webp`).
- Generate `metadata/images.parquet` with columns:
  - `image_id` (filename stem)
  - `filepath` (absolute path)
  - `file_name`
  - `site_id`
  - `width`
  - `height`
- Validate no duplicate `image_id`.

## 9) Normalization pipeline requirements
Input:
- Raw exported annotation JSON (`ann_vXXX.json`).

Output:
- `images_labels.parquet` columns:
  - `annotation_version`
  - `image_id`
  - `image_status`

- `birds.parquet` columns:
  - `annotation_version`
  - `image_id`
  - `bird_id`
  - `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h` (normalized)
  - optional px columns: `bbox_x_px`, `bbox_y_px`, `bbox_w_px`, `bbox_h_px`
  - `readability`
  - `specie`
  - `behavior`
  - `substrate`
  - `legs`

Deterministic normalization rules:
- Default initialization per region:
  - `readability=readable`
  - `specie=correct`
- Valid domains:
  - readability: `readable|occluded|unreadable`
  - specie: `correct|incorrect|unsure`
  - behavior: `flying|foraging|resting|backresting|preening|display|unsure`
  - substrate: `ground|water|air|unsure`
  - legs: `one|two|unsure|sitting`
- Mask irrelevant fields:
  - if `readability=unreadable` OR `specie=incorrect`, set `behavior/substrate/legs=null`
- Else:
  - if missing, default `behavior=unsure`, `substrate=unsure`
  - if `behavior in {resting, backresting}` and `substrate in {ground, water}`:
    - missing `legs` -> `legs=unsure`
  - otherwise `legs=null`
- Deterministic sort:
  - images sorted by (`annotation_version`, `image_id`)
  - birds sorted by (`annotation_version`, `image_id`, `bird_id`)

## 10) Crop generation requirements
Input:
- `birds.parquet` + metadata filepaths.

Behavior:
- Expand bbox around center by margin factor (default 1.2).
- Clip crop rectangle to image bounds.
- Save JPEG crops in `derived/crops/<ann_vXXX>/`.
- Save `_crops.parquet` manifest with status (`written`/`exists`) and crop dimensions.

## 11) Dataset builder requirements
Input joins:
- normalized birds
- normalized image labels
- metadata/images

Output:
- `train.parquet`, `val.parquet`, `test.parquet`
- `manifest.json`

Deterministic split policy:
- Compute bucket: `abs(hash(coalesce(site_id,'') || ':' || image_id)) % 100`
- Default split: 80/10/10
- Enforce no image leakage between splits.

Dataset row columns include at minimum:
- annotation and image identifiers
- bbox and class labels
- image_status
- filepath/site_id
- `crop_path`
- split marker

Manifest must include:
- source annotation version
- input source paths
- split formula and percentages
- row/image counts by split
- label distributions (including null counts as `<null>`)

## 12) Model B training specification (multi-head attributes)

### 12.1 Backbone and heads
Default backbone: `convnextv2_small`.

Head class counts:
- readability: 3
- specie: 3
- behavior: 7
- substrate: 4
- legs: 4

### 12.2 Label maps
- readability: `readable=0`, `occluded=1`, `unreadable=2`
- specie: `correct=0`, `incorrect=1`, `unsure=2`
- behavior: `flying=0`, `foraging=1`, `resting=2`, `backresting=3`, `preening=4`, `display=5`, `unsure=6`
- substrate: `ground=0`, `water=1`, `air=2`, `unsure=3`
- legs: `one=0`, `two=1`, `unsure=2`, `sitting=3`

### 12.3 Masking logic (critical)
Let:
- `usable = readability in {readable, occluded} AND specie != incorrect`
- `legs_relevant = usable AND behavior in {resting, backresting} AND substrate in {ground, water}`

Then:
- readability head active: always true
- specie head active: always true
- behavior head active: `usable`
- substrate head active: `usable`
- legs head active: `legs_relevant`

Use masked loss and masked metrics accordingly.

### 12.4 Training schedule
- Phase 1: freeze backbone, train heads.
- Phase 2: unfreeze backbone, fine-tune all layers lightly.

### 12.5 Outputs
For each run version:
- `checkpoint.pt` (weights + label maps + config metadata)
- `config.yaml`
- `metrics.json` (history and final metrics)
- `legs_confusion_matrix.csv` (4x4)

### 12.6 Metrics
Report per head on its valid masked subset:
- accuracy
- macro F1
- legs confusion matrix

## 13) Model C training specification (image_status)

### 13.1 Task
Binary classify image:
- `no_usable_birds=0`
- `has_usable_birds=1`

### 13.2 Data prep
- Merge `images_labels.parquet` with metadata by `image_id`.
- Keep only valid status labels.
- Remove rows with missing image files.
- Use deterministic split bucket formula as above.

### 13.3 Outputs
For each run version:
- `checkpoint.pt`
- `config.yaml`
- `metrics.json` with:
  - epoch history
  - test loss/accuracy/F1
  - 2x2 confusion matrix
  - class balance counts

## 14) Active learning specification

### 14.1 Inference batch step
For a pool of images:
- Run detector.
- Generate per-detection probability bundles for all attribute heads.
- Generate image-level usable-bird probability.
- Save `predictions.parquet` with:
  - `image_id`, `image_path`
  - `detection_count`, `max_detection_conf`
  - `model_c_has_usable_prob`
  - `detections_json` (per-detection bbox + per-head probability maps)

### 14.2 Selection scoring
Per image, compute components:
- `readability_uncertainty`
- `resting_low_legs_conf`
- `disagreement`
- `diversity`

Weighted total score:
- `0.40 * readability_uncertainty`
- `0.35 * resting_low_legs_conf`
- `0.15 * disagreement`
- `0.10 * diversity`

Selection strategy:
- Choose top `batch_size * (1 - random_slice_pct)` by total score.
- Add random slice from remaining pool (`random_slice_pct`, default 0.10).
- Deterministic with fixed seed.

Outputs:
- `selected_images.csv`
- `selection_report.json` (weights, counts, score summary)

## 15) Deployment and runtime requirements

### 15.1 Services
Implement compose-friendly services:
- annotation web app service
- annotation API service
- ML backend service
- optional postgres service for persistent state

### 15.2 Environment variables (minimum)
- `BIRDS_DATA_ROOT`
- `MODEL_A_WEIGHTS`
- `MODEL_B_CHECKPOINT`
- `MODEL_C_CHECKPOINT`
- `MODEL_A_DEVICE` (`auto|cpu|mps|cuda index`)
- detector params: `MODEL_A_CONF`, `MODEL_A_IOU`, `MODEL_A_IMGSZ`, `MODEL_A_MAX_DET`
- `BIRDS_LOG_LEVEL`

### 15.3 Cross-platform device resolution
- `auto` preference order:
  1) CUDA (if available)
  2) MPS (if available)
  3) CPU
- Must not crash if requested accelerator is unavailable; fallback to CPU with warning.

## 16) Benchmark and observability
- Add backend latency benchmark over sample images.
- Export per-sample CSV and summary JSON (min/mean/p50/p95/max, success/error counts).
- Log per-task inference summary and prediction envelope shape for debugging.

## 17) Testing and acceptance criteria
Provide automated tests that cover:

1. Normalization
- Correct masking/defaults.
- Support for region choice linking variants.
- Domain validation of categorical values.

2. Training masks
- Unit tests for unreadable/specie-incorrect gating.
- Legs mask only active in valid subspace.

3. Prediction contract
- `/predict` returns valid envelope and result objects.
- Conditional emission of nested fields works.

4. Determinism
- Same annotation export -> identical normalized outputs.
- Same input and seed -> identical dataset split assignment and AL selection.

5. End-to-end smoke
- Annotate sample tasks in custom app.
- Export `ann_vNNN`.
- Normalize -> crops -> dataset -> train B/C -> run prediction service.

Acceptance is complete when all above pass and artifact lineage is versioned with no overwrite.

## 18) Deliverables expected from implementation LLM
- Full source code for annotation frontend and backend (no Label Studio).
- ML backend service and predictor adapters.
- Data pipeline scripts for register/normalize/crops/dataset/train/infer/select.
- Config files, compose setup, and environment docs.
- Test suite (unit + integration + smoke).
- Runbook documenting full reproducible workflow.

## 19) Explicit non-goals for v2
- Detector fine-tuning.
- Distributed training orchestration.
- Multi-tenant auth complexity beyond what is needed for <=3 annotators.
- Cloud object storage integration.

