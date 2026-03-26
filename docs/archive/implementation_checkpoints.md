# Implementation Checkpoints

## C0 Environment contract
- Implemented:
  - `/Users/antoine/bird_leg/ENVIRONMENT.md`
  - `/Users/antoine/bird_leg/.env.example`
  - `/Users/antoine/bird_leg/pyproject.toml`
  - `/Users/antoine/bird_leg/Makefile`
  - `/Users/antoine/bird_leg/uv.lock`
- Verified:
  - `docker compose --env-file /Users/antoine/bird_leg/.env ... config` passes
  - `ENVIRONMENT.md` includes required env vars and checks

## C1 Scaffold
- Implemented:
  - `/Users/antoine/bird_leg/src/birdsys/*`
  - `/Users/antoine/bird_leg/scripts/*`
  - `/Users/antoine/bird_leg/config/project.example.yaml`
  - `/Users/antoine/bird_leg/tests/smoke/test_imports.py`

## C2 Data root + metadata
- Implemented:
  - `/Users/antoine/bird_leg/scripts/register_images.py`
  - local fallback data layout at `/Users/antoine/bird_leg/data/birds_project`
- Blocked in this environment:
  - `/data` mount is read-only, cannot create `/data/birds_project`
  - pandas/pyarrow/duckdb unavailable, so metadata parquet generation cannot be executed here

## C3 Label Studio config
- Implemented:
  - `/Users/antoine/bird_leg/deploy/docker-compose.yml`
  - `/Users/antoine/bird_leg/labelstudio/label_config.xml`
  - `/Users/antoine/bird_leg/docs/annotator_guide.md`
- Verified:
  - compose config is valid
  - XML is well-formed (`xmllint --noout`)

## C4-C10 Pipelines and backend
- Implemented:
  - ML backend service + predictors
  - normalization/crops/dataset scripts
  - training scripts for Model B and Model C
  - batch inference + active-learning selection scripts
  - SQL template for DuckDB dataset build

## C11 Repro + integration docs
- Implemented:
  - `/Users/antoine/bird_leg/docs/runbook_e2e.md`
  - `/Users/antoine/bird_leg/docs/reproducibility_checklist.md`
  - integration tests in `/Users/antoine/bird_leg/tests/integration`
