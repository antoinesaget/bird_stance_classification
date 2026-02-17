# BirdSys Environment Contract

## Scope
This file defines the reproducible runtime contract for the Bird Annotation & Classification System on:
- Linux + NVIDIA RTX 3090 (primary training/inference)
- macOS + Apple Silicon M4 (development + light inference)

## Baseline
- Python: `3.11.x`
- Dependency manager: [`uv`](https://github.com/astral-sh/uv)
- Runtime split:
  - Docker Compose: Label Studio + Postgres + ML backend service
  - Host `uv` environment: data pipelines, dataset builds, training jobs
- Canonical data root: `/data/birds_project`

If `/data` is unavailable (read-only or restricted), use a local override such as:
`/Users/antoine/bird_leg/data/birds_project` and set `BIRDS_DATA_ROOT` accordingly.

## Required Environment Variables
- `BIRDS_DATA_ROOT` (default: `/data/birds_project`)
- `LABEL_STUDIO_URL` (example: `http://localhost:8080`)
- `LABEL_STUDIO_API_TOKEN` (Label Studio personal token)
- `MODEL_A_WEIGHTS` (YOLO checkpoint path, example: `/Users/antoine/bird_leg/yolo11m.pt`)
- `MODEL_B_CHECKPOINT` (Model B checkpoint path)
- `MODEL_C_CHECKPOINT` (Model C checkpoint path)

## Toolchain Requirements
- Docker + Docker Compose v2
- Python 3.11 installed and discoverable
- `uv` installed and available in PATH

## Bootstrap
```bash
cd /Users/antoine/bird_leg
cp .env.example .env
make bootstrap
```

## Verification Commands
### Python / uv
```bash
uv run python -V
```
Expected: `Python 3.11.x`

### Compose syntax
```bash
docker compose -f /Users/antoine/bird_leg/deploy/docker-compose.yml config
```
Expected: valid rendered compose config

### macOS MPS check
```bash
uv run python -c "import torch; print(torch.backends.mps.is_available())"
```
Expected on macOS M4: `True` when torch build exposes MPS

### Linux CUDA check
```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```
Expected on RTX 3090 host: `True` with CUDA-enabled torch runtime

## Storage & Reproducibility Rules
- Never mutate raw image files in `${BIRDS_DATA_ROOT}/raw_images`.
- Never overwrite prior annotation exports/normalized outputs/dataset versions/model versions.
- Rebuild inputs of record:
  - `${BIRDS_DATA_ROOT}/raw_images`
  - `${BIRDS_DATA_ROOT}/metadata/images.parquet`
  - `${BIRDS_DATA_ROOT}/labelstudio/exports/ann_vXXX.json`
  - code + config in this repository

## Path Layout
Expected top-level data layout under `${BIRDS_DATA_ROOT}`:
- `raw_images/`
- `metadata/`
- `labelstudio/exports/`
- `labelstudio/normalized/`
- `derived/crops/`
- `derived/datasets/`
- `models/detector/`
- `models/attributes/`
- `models/image_status/`
