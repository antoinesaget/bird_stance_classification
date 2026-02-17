SHELL := /bin/zsh

UV ?= uv
PYTHON_VERSION ?= 3.11
REPO_ROOT ?= /Users/antoine/bird_leg
COMPOSE_FILE ?= /Users/antoine/bird_leg/deploy/docker-compose.yml
ENV_FILE ?= $(REPO_ROOT)/.env
COMPOSE ?= docker compose --env-file $(ENV_FILE) -f $(COMPOSE_FILE)
ML_BACKEND_HOST_PORT ?= 9091

.PHONY: bootstrap check test smoke compose-config compose-up compose-down compose-ps run-ml-backend run-ml-backend-host stop-ml-backend-container

bootstrap:
	$(UV) sync --python $(PYTHON_VERSION)

check:
	$(UV) run python -V
	$(UV) run pytest -q tests/smoke/test_imports.py

smoke:
	$(UV) run pytest -q tests/smoke/test_imports.py

compose-config:
	$(COMPOSE) config >/dev/null
	@echo "compose config valid"

compose-up:
	$(COMPOSE) up -d

compose-down:
	$(COMPOSE) down

compose-ps:
	$(COMPOSE) ps

run-ml-backend:
	$(UV) run uvicorn services.ml_backend.app.main:app --host 0.0.0.0 --port 9090 --reload

run-ml-backend-host:
	set -a; source $(ENV_FILE); set +a; \
	MODEL_A_DEVICE=$${MODEL_A_DEVICE:-auto} \
	$(UV) run uvicorn services.ml_backend.app.main:app --host 0.0.0.0 --port $(ML_BACKEND_HOST_PORT) --reload

stop-ml-backend-container:
	$(COMPOSE) stop ml-backend

test:
	$(UV) run pytest -q
