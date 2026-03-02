# BirdLeg Deployment (TrueNAS App Stack + External ML Backend)

This guide is for deploying **everything except ML inference** on TrueNAS:
- Label Studio (UI + API)
- PostgreSQL (Label Studio DB)
- local image/data storage for annotation tasks

The ML backend stays on a separate machine and is connected from Label Studio by URL.

## 1) Is The Project Already Set Up For This?

Short answer: **yes, with one recommended compose file**.

- Label Studio already supports an external ML backend URL.
- The repository now includes an app-only compose file:
  - `deploy/docker-compose.app-only.yml`
- This avoids the default `deploy/docker-compose.yml` dependency on the in-repo `ml-backend` service and avoids `MODEL_A_WEIGHTS` mount issues when you do not run ML in that stack.

## 2) Target Architecture

- TrueNAS server:
  - `birds-postgres` container
  - `birds-label-studio` container
  - mounted dataset containing BirdLeg data (`BIRDS_DATA_ROOT`)
- External ML host (existing machine):
  - runs `services/ml_backend` on private network (example `http://<ml-host>:9090`)
- Label Studio calls ML host over network; annotators never call ML directly.

## 3) What Must Exist In TrueNAS Storage

Pick one TrueNAS dataset root (example below):

- `/mnt/<pool>/apps/bird_leg/repo` (git clone)
- `/mnt/<pool>/apps/bird_leg/data/birds_project` (`BIRDS_DATA_ROOT`)
- `/mnt/<pool>/apps/bird_leg/postgres` (optional bind mount for DB files)
- `/mnt/<pool>/apps/bird_leg/labelstudio_data` (optional bind mount for LS app data)
- `/mnt/<pool>/apps/bird_leg/backups` (optional backups)

`BIRDS_DATA_ROOT` should contain at least:

- `raw_images/<site_id>/...`
- `labelstudio/imports/` (task json files)
- `labelstudio/exports/` (annotation exports)
- optional compressed mirrors:
  - `labelstudio/images_compressed/q35/<site_id>/...`
  - `labelstudio/images_compressed/q60/<site_id>/...`

## 4) Clone And Prepare Repo On TrueNAS

```bash
cd /mnt/<pool>/apps/bird_leg
git clone <YOUR_GITHUB_REPO_URL> repo
cd repo
```

Create a deployment env file dedicated to TrueNAS:

```bash
cat > .env.truenas <<'EOF'
# Core
BIRDS_DATA_ROOT=/mnt/<pool>/apps/bird_leg/data/birds_project

# Label Studio
LABEL_STUDIO_PORT=8080
LABEL_STUDIO_HOST=https://<your-public-labelstudio-domain>
LABEL_STUDIO_USERNAME=admin@local
LABEL_STUDIO_PASSWORD=<strong-password>

# Optional: use bind mounts instead of docker named volumes
LABEL_STUDIO_PGDATA_DIR=/mnt/<pool>/apps/bird_leg/postgres
LABEL_STUDIO_APP_DATA_DIR=/mnt/<pool>/apps/bird_leg/labelstudio_data
EOF
```

Create directories:

```bash
mkdir -p /mnt/<pool>/apps/bird_leg/data/birds_project
mkdir -p /mnt/<pool>/apps/bird_leg/postgres
mkdir -p /mnt/<pool>/apps/bird_leg/labelstudio_data
mkdir -p /mnt/<pool>/apps/bird_leg/backups
```

## 5) Start App-Only Stack (No ML Container)

```bash
cd /mnt/<pool>/apps/bird_leg/repo
docker compose --env-file .env.truenas -f deploy/docker-compose.app-only.yml up -d
docker compose --env-file .env.truenas -f deploy/docker-compose.app-only.yml ps
```

Expected containers:

- `birds-postgres` (healthy)
- `birds-label-studio` (up)

Check logs:

```bash
docker logs --tail 200 birds-postgres
docker logs --tail 200 birds-label-studio
```

## 6) Migrate Existing App Data From Current Server (Optional But Typical)

If you are moving an existing project from another host (for example `iats`), migrate:

1. PostgreSQL DB
2. Label Studio app data directory/volume
3. `BIRDS_DATA_ROOT`

### 6.1 PostgreSQL Dump On Source

```bash
docker exec birds-postgres pg_dump -U labelstudio -d labelstudio > /tmp/labelstudio.sql
```

Copy to TrueNAS:

```bash
scp /tmp/labelstudio.sql <truenas-user>@<truenas-host>:/mnt/<pool>/apps/bird_leg/backups/
```

### 6.2 Restore On TrueNAS

Stop Label Studio temporarily:

```bash
cd /mnt/<pool>/apps/bird_leg/repo
docker compose --env-file .env.truenas -f deploy/docker-compose.app-only.yml stop label-studio
```

Restore dump:

```bash
cat /mnt/<pool>/apps/bird_leg/backups/labelstudio.sql | docker exec -i birds-postgres psql -U labelstudio -d labelstudio
```

Restart Label Studio:

```bash
docker compose --env-file .env.truenas -f deploy/docker-compose.app-only.yml start label-studio
```

### 6.3 Copy `BIRDS_DATA_ROOT`

Use `rsync` from source host to TrueNAS:

```bash
rsync -av --progress <source-host>:/path/to/birds_project/ /mnt/<pool>/apps/bird_leg/data/birds_project/
```

## 7) Configure Label Studio For Local Files (Critical)

`/data/local-files/?d=...` works only if each project has a Local Files storage path that prefixes your task paths.

For mixed raw + compressed paths, set project local storage path to:

- `/data/birds_project`

Do this in Label Studio UI:

1. Project `Settings`
2. `Cloud Storage` / `Storage` (Local files)
3. Path: `/data/birds_project`
4. Save and sync if needed

If path is narrower (for example `/data/birds_project/raw_images/v1`), compressed task URLs will fail with 404.

## 8) Connect External ML Backend

In Label Studio project settings:

1. `Machine Learning`
2. Add backend URL:
   - `http://<ml-host>:9090` (or your TLS endpoint)
3. Click `Check connection`

Expected:

- `/health` returns `status=UP`
- `/setup` succeeds

Notes:

- ML host must be reachable from TrueNAS network.
- Prefer private VLAN/LAN; do not expose ML backend publicly.

## 9) Import Tasks (Raw Or Compressed)

Task JSON files generated by repo scripts should use:

- `/data/local-files/?d=birds_project/...`

Examples:

- raw: `birds_project/raw_images/v1/<image>.jpg`
- compressed: `birds_project/labelstudio/images_compressed/q35/v1/<image>.jpg`

Import via UI:

1. Open project
2. `Import`
3. upload `*.tasks.json`

## 10) Running Compression + Import Prep On TrueNAS

If Python/`uv` tooling is available on TrueNAS host:

```bash
cd /mnt/<pool>/apps/bird_leg/repo
uv sync --python 3.11
```

Build full compressed mirror:

```bash
uv run python scripts/build_annotation_image_mirror.py \
  --data-root "$BIRDS_DATA_ROOT" \
  --site-id v1 \
  --quality 35 \
  --max-images 0 \
  --overwrite
```

Build import sample (simple deterministic):

```bash
uv run python scripts/prepare_labelstudio_import.py \
  --data-root "$BIRDS_DATA_ROOT" \
  --site-id v1 \
  --count 500 \
  --sample-mode random \
  --seed 42 \
  --image-relative-root labelstudio/images_compressed/q35 \
  --output-prefix v1_q35_sample500
```

## 11) Health And Smoke Checks

Container/network checks:

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
curl -sS http://127.0.0.1:${LABEL_STUDIO_PORT:-8080}/health || true
```

DB check:

```bash
docker exec birds-postgres psql -U labelstudio -d labelstudio -c "SELECT id, title FROM project ORDER BY id;"
```

Local-files path check (inside container):

```bash
docker exec birds-label-studio ls -l /data/birds_project
```

## 12) Operations Runbook

Start:

```bash
docker compose --env-file .env.truenas -f deploy/docker-compose.app-only.yml up -d
```

Stop:

```bash
docker compose --env-file .env.truenas -f deploy/docker-compose.app-only.yml down
```

Upgrade app image:

```bash
docker compose --env-file .env.truenas -f deploy/docker-compose.app-only.yml pull
docker compose --env-file .env.truenas -f deploy/docker-compose.app-only.yml up -d
```

## 13) Backup Strategy

At minimum back up:

1. PostgreSQL dump (`pg_dump`)
2. `LABEL_STUDIO_APP_DATA_DIR` (or docker volume `labelstudio_data`)
3. `BIRDS_DATA_ROOT`

Recommended nightly DB backup:

```bash
ts=$(date +%Y%m%d_%H%M%S)
docker exec birds-postgres pg_dump -U labelstudio -d labelstudio > /mnt/<pool>/apps/bird_leg/backups/labelstudio_${ts}.sql
```

## 14) Known Pitfalls

1. Using `deploy/docker-compose.yml` without ML env vars:
   - may fail due to unresolved `MODEL_A_WEIGHTS` bind mount.
   - use `deploy/docker-compose.app-only.yml` on TrueNAS app stack.
2. Local files 404 for compressed images:
   - project storage path too narrow.
   - set to `/data/birds_project`.
3. ML backend unreachable:
   - check routing/firewall from TrueNAS to ML host.
4. Slow grid:
   - use compressed mirror (`q35`/`q60`) and ensure browser cache is enabled (already handled by override).

## 15) Minimal Command Sequence (Fresh Install)

```bash
cd /mnt/<pool>/apps/bird_leg
git clone <YOUR_GITHUB_REPO_URL> repo
cd repo
cat > .env.truenas <<'EOF'
BIRDS_DATA_ROOT=/mnt/<pool>/apps/bird_leg/data/birds_project
LABEL_STUDIO_PORT=8080
LABEL_STUDIO_HOST=https://<your-public-labelstudio-domain>
LABEL_STUDIO_USERNAME=admin@local
LABEL_STUDIO_PASSWORD=<strong-password>
LABEL_STUDIO_PGDATA_DIR=/mnt/<pool>/apps/bird_leg/postgres
LABEL_STUDIO_APP_DATA_DIR=/mnt/<pool>/apps/bird_leg/labelstudio_data
EOF
mkdir -p /mnt/<pool>/apps/bird_leg/data/birds_project /mnt/<pool>/apps/bird_leg/postgres /mnt/<pool>/apps/bird_leg/labelstudio_data
docker compose --env-file .env.truenas -f deploy/docker-compose.app-only.yml up -d
```
