# Repo Audit 2026-04-10

## Snapshot

- Local smoke tests pass from the checked-in virtualenv: `.venv/bin/pytest -q tests/smoke`
- Local checkout has the core ops code, but the working tree is cluttered by large untracked research and presentation directories
- `iats` is on `main` at `c1144e4`, the ML backend is healthy, and the repo root has accumulated large experimental directories:
  - `data/` about `20G`
  - `archive/` about `30G`
  - `.sandboxes/` about `9.8G`
  - `standalone/` about `7.8G`
  - `weights/` about `810M`
- TrueNAS is also on `main` at `c1144e4`, the `bird-stance-classification` app is running, and the repo checkout there is currently clean

## Keep As Core Workflow

These pieces match the workflow you said you still want:

- TrueNAS export and Label Studio task management:
  - `scripts/export_labelstudio_snapshot.py`
  - `scripts/create_labelstudio_project.py`
  - `scripts/build_labelstudio_localfiles_batch.py`
  - `scripts/import_labelstudio_tasks.py`
  - `scripts/prefill_labelstudio_predictions.py`
  - `scripts/ops/truenas_*.sh`
- Dataset normalization and versioned dataset creation:
  - `scripts/export_normalize.py`
  - `scripts/build_dataset.py`
  - `src/birdsys/paths.py`
  - `src/birdsys/reporting.py`
- Model training, evaluation, and promotion:
  - `scripts/train_attributes.py`
  - `scripts/train_attributes_cv.py`
  - `scripts/evaluate_model_b_checkpoint.py`
  - `scripts/promote_model.py`
  - `services/ml_backend/...`
  - `scripts/ops/iats_*.sh`

## Archive Or Remove Next

These are not on the operational spine and should move out of the main repo path first:

- Generated or one-off directories:
  - `.sandboxes/`
  - `artifacts/`
  - `final_selection/`
  - `final_selection_annotated/`
  - `2026-04-08-autoresearch-birds/`
  - top-level `archive/`, `standalone/`, `runs/`, and `weights/` on `iats`
- Scripts with no current Makefile, docs, or service references:
  - `scripts/build_annotation_image_mirror.py`
  - `scripts/infer_batch.py`
  - `scripts/migrate_labelstudio_export_to_isbird.py`
  - `scripts/predict_resting_from_detections.py`
  - `scripts/prepare_labelstudio_import.py`
  - `scripts/register_images.py`
  - `scripts/select_active_learning_batch.py`
  - `scripts/train_image_status.py`
  - `scripts/train_model_b_specialists_cv.py`
  - `scripts/report_project7_model_b_refresh.py`
- Project-7-specific reporting code that should probably leave the main path once the refresh campaign is finished:
  - `scripts/report_project7_annotation_update.py`
  - `scripts/report_project7_specialist_review.py`
  - matching `scripts/ops/iats_report_project7_*.sh`

## Main Sources Of Brittleness

- Generated assets and long-lived experiments live under the repo root on the same hosts as production code
- The sandbox template existed only under local `.sandboxes/`, so the autoresearch setup was not reproducible from a clean checkout
- The workflow is split across many ad hoc scripts instead of one stable CLI surface
- Host-specific state is documented, but enforcement is weak:
  - the Makefile assumes `uv` in `PATH`
  - some sandbox files still encode current project-7 assumptions directly

## Changes Applied In This Pass

- moved the Model B autoresearch template into tracked source at `sandbox_templates/model_b_autoresearch_p7_v1/`
- added `scripts/create_autoresearch_sandbox.py` to create a fresh nested-git sandbox under `.sandboxes/`
- repointed the sandbox unit test at the tracked template
- expanded `.gitignore` so obvious generated and research-only directories stop polluting repo status

## Recommended Follow-Up

1. Replace the script sprawl with a single `birdsys` CLI entrypoint and keep the shell scripts as thin remote wrappers only.
2. Split `scripts/` into `core/`, `experiments/`, and `archive/` or move the experiment/report code entirely under `archive/`.
3. Add one `doctor` command that checks:
   - local toolchain
   - repo cleanliness on local, `iats`, and TrueNAS
   - ML backend health
   - TrueNAS app health
   - required env files
4. Move long-lived experiment state off the repo root on `iats`, for example under `/home/antoine/bird_experiments/`.
5. Collapse project-specific report flows once project 7 is stable, so the maintained loop is only:
   - export annotations
   - normalize
   - build dataset
   - train/evaluate
   - promote model
   - refresh predictions
   - prepare/import new tasks
