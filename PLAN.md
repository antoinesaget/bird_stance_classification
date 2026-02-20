# Implementation Plan: Bird Annotation & Classification System (Compose + `uv`, Python 3.11, `/data` root)

## Summary
This plan builds the full system in gated checkpoints, starting with an environment contract file at `/Users/antoine/bird_leg/ENVIRONMENT.md`, then progressing from data and annotation setup to model training, backend integration, and active learning.

It uses your existing corpus at `/Users/antoine/bird_leg/scraped_images/scolop2_10k` (currently 9,847 images) for smoke and early integration runs.

## Checkpoints

### Checkpoint 0: Environment contract (first deliverable)
Goal: create one source of truth for reproducible setup on macOS M4 and Linux RTX 3090.

Deliverables:
- `/Users/antoine/bird_leg/ENVIRONMENT.md`
- `/Users/antoine/bird_leg/.env.example`
- `/Users/antoine/bird_leg/pyproject.toml`
- `/Users/antoine/bird_leg/uv.lock`
- `/Users/antoine/bird_leg/Makefile` with bootstrap/check targets

`ENVIRONMENT.md` must include:
- Python baseline: `3.11.x`
- Runtime mode: Docker Compose for Label Studio + ML backend, host `uv` venv for pipelines/training
- Canonical data root: `/data/birds_project`
- Required env vars: `BIRDS_DATA_ROOT`, `LABEL_STUDIO_URL`, `LABEL_STUDIO_API_TOKEN`, `MODEL_A_WEIGHTS`, `MODEL_B_CHECKPOINT`, `MODEL_C_CHECKPOINT`
- Machine verification commands for Mac (MPS) and Linux (CUDA)

Verification:
- `uv run python -V` returns 3.11.x
- `docker compose -f /Users/antoine/bird_leg/deploy/docker-compose.yml config` validates
- Mac check: `uv run python -c "import torch; print(torch.backends.mps.is_available())"`
- Linux check: `uv run python -c "import torch; print(torch.cuda.is_available())"`

### Checkpoint 1: Repository scaffold and config contract
Goal: establish clean project boundaries separate from existing model-comparison scripts.

Deliverables:
- `/Users/antoine/bird_leg/src/birdsys/` package with modules: `config.py`, `paths.py`, `logging.py`
- `/Users/antoine/bird_leg/scripts/` entrypoints:
  - `export_normalize.py`
  - `make_crops.py`
  - `build_dataset.py`
  - `train_attributes.py`
  - `train_image_status.py`
  - `infer_batch.py`
- `/Users/antoine/bird_leg/config/project.example.yaml`
- `/Users/antoine/bird_leg/tests/smoke/test_imports.py`

Verification:
- `uv run pytest /Users/antoine/bird_leg/tests/smoke/test_imports.py`
- `uv run python /Users/antoine/bird_leg/scripts/build_dataset.py --help` for all new scripts

### Checkpoint 2: Data root initialization and metadata indexing
Goal: enforce the versioned storage layout and immutable raw image policy.

Deliverables:
- Create `/data/birds_project` directory tree from spec (`raw_images`, `metadata`, `labelstudio`, `derived`, `models`)
- Image registry script `/Users/antoine/bird_leg/scripts/register_images.py`
- `/data/birds_project/metadata/images.parquet` with one row per image

Default policy:
- `image_id` = filename stem
- `site_id` = `scolop2` for current corpus
- `filepath` stored as absolute path under `/data/birds_project/raw_images/...`

Verification:
- Registered count equals 9,847 for current `scolop2_10k` set
- No duplicate `image_id`
- Raw image files are read-only by convention and never rewritten by scripts

### Checkpoint 3: Label Studio deployment and annotation config
Goal: make annotation workflow usable for up to 3 annotators with required conditional logic.

Deliverables:
- `/Users/antoine/bird_leg/deploy/docker-compose.yml` with `label-studio`, `postgres`, `ml-backend`
- `/Users/antoine/bird_leg/labelstudio/label_config.xml` implementing:
  - required `image_status`
  - `Bird` bbox label
  - conditional visibility/requirements for readability/activity/support/standing-only fields
- `/Users/antoine/bird_leg/docs/annotator_guide.md`

Verification:
- Stack starts and Label Studio UI is reachable
- Project imports 50 sample images from `scolop2_10k`
- Manual UI test confirms conditional fields behave exactly per spec

### Checkpoint 4: ML backend baseline + Model A (YOLO pre-annotations)
Goal: provide automatic bbox proposals when opening tasks.

Deliverables:
- `/Users/antoine/bird_leg/services/ml_backend/app/main.py` (Label Studio ML backend)
- `/Users/antoine/bird_leg/services/ml_backend/app/predictors/model_a_yolo.py`
- Prediction serialization layer producing Label Studio-compatible results with confidence

Verification:
- `GET /healthz` returns healthy
- `POST /predict` returns bboxes for sample tasks
- Label Studio opening a new task shows YOLO proposals without manual API calls

### Checkpoint 5: Export normalization and crop generation
Goal: deterministic conversion from Label Studio exports to training-ready structured data.

Deliverables:
- `/Users/antoine/bird_leg/scripts/export_normalize.py` producing:
  - `/data/birds_project/labelstudio/normalized/ann_vXXX/images_labels.parquet`
  - `/data/birds_project/labelstudio/normalized/ann_vXXX/birds.parquet`
- `/Users/antoine/bird_leg/scripts/make_crops.py` writing bird crops to `/data/birds_project/derived/crops/ann_vXXX/`

Verification:
- Schema checks pass for both Parquet outputs
- Unreadable rows have nulls in masked fields exactly as specified
- Re-running normalization on same `ann_vXXX.json` yields byte-stable Parquet outputs

### Checkpoint 6: Dataset assembly with DuckDB
Goal: build versioned datasets `ds_vXXX` with deterministic splits and manifests.

Deliverables:
- `/Users/antoine/bird_leg/sql/build_dataset_duckdb.sql`
- `/Users/antoine/bird_leg/scripts/build_dataset.py`
- Output dataset folder:
  - `/data/birds_project/derived/datasets/ds_vXXX/train.parquet`
  - `/data/birds_project/derived/datasets/ds_vXXX/val.parquet`
  - `/data/birds_project/derived/datasets/ds_vXXX/test.parquet`
  - `/data/birds_project/derived/datasets/ds_vXXX/manifest.json`

Split policy:
- If `site_id`/date exist: stratified grouping to reduce leakage
- Else: stable hash split on `image_id`

Verification:
- Train/val/test are disjoint on `image_id`
- Manifest contains source `ann_vXXX`, filters, split policy, label counts
- Rebuild with same inputs gives identical row assignments

### Checkpoint 7: Model B training pipeline (multi-head attributes)
Goal: train masked multi-head classifier with ConvNeXtV2-S backbone.

Deliverables:
- `/Users/antoine/bird_leg/scripts/train_attributes.py`
- `/Users/antoine/bird_leg/config/train_attributes.yaml`
- Outputs under `/data/birds_project/models/attributes/convnextv2s_vXXX/`:
  - `config.yaml`
  - `checkpoint.pt`
  - `metrics.json`
  - `legs_confusion_matrix.csv`

Critical implementation rules:
- Apply exact masking logic from spec per head
- Two-stage schedule: frozen backbone then partial unfreeze

Verification:
- Smoke run on small subset completes and saves artifacts
- Metrics are computed on correct masked subsets only
- Unit test validates masking logic on synthetic label cases

### Checkpoint 8: Model C training pipeline (image_status)
Goal: train image-level binary classifier for `has_usable_birds` vs `no_usable_birds`.

Deliverables:
- `/Users/antoine/bird_leg/scripts/train_image_status.py`
- `/Users/antoine/bird_leg/config/train_image_status.yaml`
- Outputs under `/data/birds_project/models/image_status/status_vXXX/`

Verification:
- Smoke run writes checkpoint + metrics
- Metrics include class balance, confusion matrix, F1
- Inference script can score an image batch and return calibrated probabilities

### Checkpoint 9: Backend integration of Model B + Model C
Goal: return full prediction bundle (bbox + attributes + image_status) in one backend response.

Deliverables:
- `/Users/antoine/bird_leg/services/ml_backend/app/predictors/model_b_attributes.py`
- `/Users/antoine/bird_leg/services/ml_backend/app/predictors/model_c_image_status.py`
- Unified prediction orchestrator in backend

Verification:
- Single `/predict` call runs A+B+C and returns confidence per field
- Label Studio task opening shows prefilled image_status and object attributes
- Latency smoke benchmark logged for 20 images

### Checkpoint 10: Manual active learning pipeline
Goal: generate high-value annotation batches from Model B uncertainty.

Deliverables:
- `/Users/antoine/bird_leg/scripts/infer_batch.py` for pooled inference
- `/Users/antoine/bird_leg/scripts/select_active_learning_batch.py`
- Selection outputs:
  - `selected_images.csv`
  - `selection_report.json` with heuristic contributions

Heuristics to implement:
- readability uncertainty near 0.5
- standing+readable with low legs confidence
- disagreement pattern score
- diversity term + random exploration slice

Verification:
- Running on unlabeled pool produces deterministic selection with fixed seed
- Report includes score breakdown per selected image
- Batch can be imported into Label Studio without manual edits

### Checkpoint 11: End-to-end reproducibility and release gate
Goal: prove full loop works with version lineage and no overwrites.

Deliverables:
- `/Users/antoine/bird_leg/docs/runbook_e2e.md`
- `/Users/antoine/bird_leg/docs/reproducibility_checklist.md`
- Integration test suite under `/Users/antoine/bird_leg/tests/integration/`

Verification:
- End-to-end dry run:
  1) annotate sample
  2) export `ann_vNNN`
  3) normalize
  4) build `ds_vNNN`
  5) train `convnextv2s_vNNN` + `status_vNNN`
  6) serve predictions in Label Studio
- All artifacts are versioned; no previous version is overwritten
- Dataset/model manifests link back to source annotation version

## Important Public APIs / Interfaces / Types
- Config interface via `/Users/antoine/bird_leg/config/project.example.yaml` and env vars in `/Users/antoine/bird_leg/.env.example`.
- ML backend HTTP interface:
  - `GET /healthz`
  - `POST /predict` (Label Studio task payload in, prediction bundle out).
- Normalized data schemas:
  - `images_labels.parquet`: `annotation_version`, `image_id`, `image_status`
  - `birds.parquet`: `annotation_version`, `image_id`, `bird_id`, bbox normalized/pixel coords, `readability`, `activity`, `support`, `legs`, `resting_back`
- Dataset manifest schema in `manifest.json`: source annotation version, split policy, filters, label counts, generation timestamp, code version.
- Training outputs contract for Model B/C folders: `config.yaml`, `checkpoint.pt`, `metrics.json`.

## Test Cases and Scenarios
- Schema unit tests: verify Parquet columns/types/nullability and masking behavior.
- Determinism tests: same input export yields identical normalized outputs and dataset splits.
- Backend contract tests: valid Label Studio prediction JSON with confidences and expected fields.
- Model pipeline smoke tests: short train/infer cycle for Model B and Model C on sampled data.
- Cross-platform checks: Mac M4 path (MPS where available) and Linux 3090 path (CUDA) both execute core commands.
- End-to-end integration test: annotation to retraining to serving predictions in one scripted flow.

## Assumptions and Defaults Chosen
- Dependency manager: `uv` with lockfile.
- Python baseline: `3.11.x`.
- Runtime split: Docker Compose for Label Studio + ML backend; host venv for pipelines/training.
- Canonical data root: `/data/birds_project`.
- Existing folder `/Users/antoine/bird_leg/shared/scripts` remains as legacy utilities and is not the main production pipeline path.
- Initial corpus for implementation/smoke: `/Users/antoine/bird_leg/scraped_images/scolop2_10k` (9,847 files currently present).
- No MinIO/S3, no SAM refinement, no YOLO fine-tuning in this phase.
