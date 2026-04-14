# ML Backend

This subproject is the remaining served inference surface for BirdSys.

## Surviving Files

- `src/birdsys/ml_backend/app/main.py`: FastAPI app with `/healthz`, `/health`, `/setup`, `/validate`, and `/predict`
- `src/birdsys/ml_backend/app/predictors/model_a_yolo.py`: Model A detector adapter
- `src/birdsys/ml_backend/app/predictors/model_b_attributes.py`: Model B attribute inference adapter with artifact and heuristic modes
- `src/birdsys/ml_backend/app/serializers.py`: Label Studio prediction serialization
- `src/birdsys/ml_backend/app/response_contract.py`: response shaping helpers
- `src/birdsys/ml_backend/promote_model.py`: promote a candidate artifact into the served release layout
- `Dockerfile`: container build for the backend
- `deploy/docker-compose.ai-ml.yml`: live `ai` deployment
- `deploy/env/ai.env.example`: example environment for the `ai` deploy

## Runtime Shape

- Model A is a YOLO detector.
- Model B is an attribute predictor that can load an artifact bundle or fall back to heuristics.
- The backend is meant to serve Label Studio-compatible predictions on port `9090`.
- `ai` is the intended live host for this service.

## Current Shape

- The old `cli.py` wrapper is gone.
- The remaining surface is the FastAPI app, the predictor modules, the promotion helper, and the Docker deploy files.

## Current Status

- `app/main.py` imports `birdsys.core` for environment loading, logging, and health payload contracts.
- `promote_model.py` imports `birdsys.core.PromotionMetadata`.
- Because the shared core package still references deleted modules, this subproject is not currently bootable from the cleaned repo.
