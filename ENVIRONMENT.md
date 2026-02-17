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
- `MODEL_A_WEIGHTS` (host checkpoint path for local scripts, example: `/Users/antoine/bird_leg/yolo11m.pt`)
- `MODEL_A_DEVICE` (`auto`, `cpu`, `mps`, or CUDA index like `0`; default `auto`)
- `MODEL_A_IMGSZ` (default `1280`)
- `MODEL_A_MAX_DET` (default `300`)
- `MODEL_A_CONF` (default `0.25`)
- `MODEL_A_IOU` (default `0.45`)
- `MODEL_B_CHECKPOINT` (Model B checkpoint path)
- `MODEL_C_CHECKPOINT` (Model C checkpoint path)

Compose binds `MODEL_A_WEIGHTS` into the ML backend container at `/models/model_a/weights.pt`.

## Optional host ML backend mode (for Apple Silicon acceleration)
Use this mode when you want Label Studio (Docker) to call an ML backend running on the host `uv` environment.

1. Stop the containerized backend:
```bash
make stop-ml-backend-container
```
2. Start host backend:
```bash
MODEL_A_DEVICE=auto make run-ml-backend-host
```
3. In Label Studio ML settings, set backend URL to:
`http://host.docker.internal:9091`

Notes:
- On macOS Docker Desktop, `host.docker.internal` resolves to the host.
- `auto` selects CUDA first, then MPS, then CPU.

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
docker compose --env-file /Users/antoine/bird_leg/.env -f /Users/antoine/bird_leg/deploy/docker-compose.yml config
```
Expected: valid rendered compose config

### macOS MPS check
```bash
uv run python -c "import torch; print(torch.backends.mps.is_available())"
```
Expected on macOS M4: `True` when torch build exposes MPS

### Backend device check
```bash
curl -sS http://127.0.0.1:9091/health
```
Expected: `"model_a_device"` reflects selected runtime (`mps`, `0`, or `cpu`)

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
