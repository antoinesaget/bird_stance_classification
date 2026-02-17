# Implementation Progress (vs `PLAN.md`)

Last updated: 2026-02-17

## Update policy
- This file is the running source of truth for implementation state vs `/Users/antoine/bird_leg/PLAN.md`.
- It must be updated on every meaningful implementation step (code, infra, verification, or checkpoint closure).
- Status values:
  - `DONE`: implemented and verified against checkpoint intent.
  - `PARTIAL`: implemented but one or more required verification items still pending.
  - `PENDING`: not yet implemented.

## Snapshot summary
- `DONE`: C0, C1, C2, C5, C6, C7, C8, C10
- `PARTIAL`: C3, C4, C9, C11
- `PENDING`: none

## Checkpoint status
| Checkpoint | Status | Evidence | Remaining |
|---|---|---|---|
| C0 Environment contract | DONE | `/Users/antoine/bird_leg/ENVIRONMENT.md`, `/Users/antoine/bird_leg/.env.example`, `/Users/antoine/bird_leg/pyproject.toml`, `/Users/antoine/bird_leg/uv.lock`, `/Users/antoine/bird_leg/Makefile`; `uv run python -V` -> `3.11.14`; compose config validates | None |
| C1 Scaffold + config contract | DONE | `/Users/antoine/bird_leg/src/birdsys/*`, `/Users/antoine/bird_leg/scripts/*`, `/Users/antoine/bird_leg/config/project.example.yaml`, `/Users/antoine/bird_leg/tests/smoke/test_imports.py`; script `--help` checks pass | None |
| C2 Data root + metadata indexing | DONE | Data root initialized at `/Users/antoine/bird_leg/data/birds_project` (container-mapped to `/data/birds_project`); `/Users/antoine/bird_leg/scripts/register_images.py`; `/Users/antoine/bird_leg/data/birds_project/metadata/images.parquet` with `rows=9847`, `duplicates=0` | None |
| C3 Label Studio deployment + annotation config | PARTIAL | Stack up (`postgres`, `label-studio`, `ml-backend`); `/Users/antoine/bird_leg/labelstudio/label_config.xml`; `/Users/antoine/bird_leg/docs/annotator_guide.md`; 50-image import bundle generated at `/Users/antoine/bird_leg/data/birds_project/labelstudio/imports/scolop2_sample50.tasks.json` | Manual Label Studio UI check still required for conditional logic behavior |
| C4 ML backend baseline + Model A | PARTIAL | `/Users/antoine/bird_leg/services/ml_backend/app/main.py`; `/Users/antoine/bird_leg/services/ml_backend/app/predictors/model_a_yolo.py`; `GET /healthz` -> `model_a_loaded=true`; `/predict` returns bbox + confidence | Manual validation still required: opening task in Label Studio auto-shows proposals without manual API calls |
| C5 Export normalization + crops | DONE | `/Users/antoine/bird_leg/scripts/export_normalize.py`, `/Users/antoine/bird_leg/scripts/make_crops.py`; outputs under `/Users/antoine/bird_leg/data/birds_project/labelstudio/normalized/ann_v900/` and `/Users/antoine/bird_leg/data/birds_project/derived/crops/ann_v900/`; deterministic rerun check (same `ann_v` path, same SHA256) | None |
| C6 Dataset assembly (DuckDB) | DONE | `/Users/antoine/bird_leg/sql/build_dataset_duckdb.sql`, `/Users/antoine/bird_leg/scripts/build_dataset.py`; datasets at `/Users/antoine/bird_leg/data/birds_project/derived/datasets/ds_v900/` and `/Users/antoine/bird_leg/data/birds_project/derived/datasets/ds_v901/`; split assignment equality check across versions (`train/val/test all_same=True`) | None |
| C7 Model B training pipeline | DONE | `/Users/antoine/bird_leg/scripts/train_attributes.py`, `/Users/antoine/bird_leg/config/train_attributes.yaml`; smoke artifacts at `/Users/antoine/bird_leg/data/birds_project/models/attributes/convnextv2s_v900_smoke/`; masking unit tests in `/Users/antoine/bird_leg/tests/unit/test_masking_rules.py` | None |
| C8 Model C training pipeline | DONE | `/Users/antoine/bird_leg/scripts/train_image_status.py`, `/Users/antoine/bird_leg/config/train_image_status.yaml`; smoke artifacts at `/Users/antoine/bird_leg/data/birds_project/models/image_status/status_v900_smoke/` | None |
| C9 Backend integration A+B+C | PARTIAL | Unified orchestration active in `/Users/antoine/bird_leg/services/ml_backend/app/main.py`; single `/predict` returns bbox + attributes + image_status; 20-image latency benchmark added and run via `/Users/antoine/bird_leg/services/ml_backend/app/benchmark_predict.py` with report at `/Users/antoine/bird_leg/data/birds_project/derived/benchmarks/ml_backend/predict_latency_20260217_113548/latency_report.json` | Manual Label Studio validation still required: fields prefilled in UI when opening task |
| C10 Active learning batch selection | DONE | `/Users/antoine/bird_leg/scripts/infer_batch.py`, `/Users/antoine/bird_leg/scripts/select_active_learning_batch.py`; outputs at `/Users/antoine/bird_leg/data/birds_project/derived/active_learning_infer/run_20260217_115653/`; deterministic selection fixed (stable hash) and verified with repeated seed run | None |
| C11 End-to-end reproducibility gate | PARTIAL | `/Users/antoine/bird_leg/docs/runbook_e2e.md`, `/Users/antoine/bird_leg/docs/reproducibility_checklist.md`, integration tests in `/Users/antoine/bird_leg/tests/integration/` | Full live annotation-driven E2E gate (annotate/export/retrain/serve) still requires manual annotation pass |

## Verification log (latest)
- `uv run pytest -q` -> `7 passed`
- `docker compose ... ps` -> all services up
- `ml-backend /healthz` -> healthy and `model_a_loaded=true`
- `ml-backend /predict` smoke -> success on sample image (`has_error=False`)
- Latency smoke (20 images): mean `2853.7049 ms`, p95 `3430.6038 ms`, max `4263.5895 ms`
