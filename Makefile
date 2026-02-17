SHELL := /bin/zsh

UV ?= uv
PYTHON_VERSION ?= 3.11
REPO_ROOT ?= /Users/antoine/bird_leg
COMPOSE_FILE ?= /Users/antoine/bird_leg/deploy/docker-compose.yml
ENV_FILE ?= $(REPO_ROOT)/.env
COMPOSE ?= docker compose --env-file $(ENV_FILE) -f $(COMPOSE_FILE)

.PHONY: bootstrap check test smoke compose-config compose-up compose-down compose-ps run-ml-backend

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

test:
	$(UV) run pytest -q
