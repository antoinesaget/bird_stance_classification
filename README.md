# BirdLeg Deployment Guide (Ubuntu + RTX 3090)

This repo is set up for:
- Label Studio + Postgres in Docker
- ML backend on host (GPU via CUDA)
- Data/images copied separately (external SSD)
- Annotation schema v2 (`isbird` per bbox, no manual `image_status`)

For a full TrueNAS-focused deployment (app stack on TrueNAS, external ML backend), see:
- `DEPLOYMENT.md`

## 1) Prepare GitHub repo (one-time)

`gh` CLI is not installed in this workspace, so use either GitHub web UI or git commands below.

Create an empty GitHub repo, then from this project root:

```bash
git remote add origin <YOUR_GITHUB_REPO_URL>
git branch -M main
git add .
git commit -m "Initial deployment-ready setup"
git push -u origin main
```

On Ubuntu:

```bash
git clone <YOUR_GITHUB_REPO_URL> bird_leg
cd bird_leg
```

## 2) Copy heavy assets from external SSD

The repository excludes datasets and weights (`.pt`, raw images, outputs).

Copy to your Ubuntu clone:

```bash
mkdir -p data/birds_project/raw_images
cp /media/$USER/<SSD_NAME>/yolo11m.pt .
rsync -a /media/$USER/<SSD_NAME>/birds_project/raw_images/ data/birds_project/raw_images/
```

Adjust paths to match your SSD layout.

## 3) Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
BIRDS_DATA_ROOT=/home/<you>/bird_leg/data/birds_project
MODEL_A_WEIGHTS=/home/<you>/bird_leg/yolo11m.pt
MODEL_A_DEVICE=0
LABEL_STUDIO_URL=http://localhost:8080
LABEL_STUDIO_USERNAME=admin@local
LABEL_STUDIO_PASSWORD=<strong-admin-password>
```

## 4) Start Docker services

```bash
docker compose --env-file .env -f deploy/docker-compose.yml up -d postgres label-studio
```

The compose file includes:
- env-driven Label Studio credentials (no hardcoded `admin/admin`)
- Linux `host.docker.internal` mapping via `extra_hosts`

## 5) Start ML backend on host (GPU)

```bash
uv sync --python 3.11
set -a; source .env; set +a
MODEL_A_DEVICE=0 uv run uvicorn services.ml_backend.app.main:app --host 0.0.0.0 --port 9091
```

Checks:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
curl -sS http://127.0.0.1:9091/health
```

In Label Studio, set ML backend URL to:

```text
http://host.docker.internal:9091
```

## 5.1) Optional migration of old exports to schema v2

```bash
uv run python scripts/migrate_labelstudio_export_to_isbird.py \
  --input-json project-1-at-2026-03-02-14-48-0edc627b.json \
  --output-json "$BIRDS_DATA_ROOT/labelstudio/exports/ann_v001_migrated_isbird.json" \
  --overwrite
```

## 5.2) Optional compressed image mirror (same resolution, q=60)

```bash
uv run python scripts/build_annotation_image_mirror.py \
  --data-root "$BIRDS_DATA_ROOT" \
  --site-id scolop2 \
  --quality 60 \
  --max-images 1000
```

Then generate imports pointing to compressed images:

```bash
uv run python scripts/prepare_labelstudio_import.py \
  --data-root "$BIRDS_DATA_ROOT" \
  --site-id scolop2 \
  --count 1000 \
  --sample-mode first \
  --image-relative-root labelstudio/images_compressed/q60
```

## 6) Rotate admin password + create Adrien account

Run after Label Studio container is up:

```bash
LABEL_STUDIO_ADMIN_PASSWORD='<strong-admin-password>' \
LABEL_STUDIO_ADRIEN_PASSWORD='<strong-adrien-password>' \
LABEL_STUDIO_ADRIEN_EMAIL='adrien@local' \
LABEL_STUDIO_ADRIEN_USERNAME='adrien' \
./scripts/bootstrap_labelstudio_users.sh
```

This script is idempotent:
- sets/rotates admin credentials
- creates or updates Adrien account

## 7) Share with your biologist friend

Expose Label Studio only (port 8080):

```bash
ngrok http --basic-auth="annotator:<very-strong-password>" 8080
```

Do not expose ML backend ports (`9090`, `9091`) publicly.

## Notes on performance

- `ngrok` works but adds latency and bandwidth caps (especially on free plan).
- For heavy image annotation, a small VPS reverse proxy (Caddy/Nginx + HTTPS) pointing to your home endpoint is usually smoother and more stable than free ngrok.
- If you want zero client install for your friend, Cloudflare Tunnel is often the best compromise (better stability than ngrok free, browser-only access for annotators).
