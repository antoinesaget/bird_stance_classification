SHELL := /bin/zsh

UV ?= uv
PYTHON_VERSION ?= 3.11
COMPOSE_FILE ?= /Users/antoine/bird_leg/deploy/docker-compose.yml

.PHONY: bootstrap check test smoke compose-config run-ml-backend

bootstrap:
	$(UV) sync --python $(PYTHON_VERSION)

check:
	$(UV) run python -V
	$(UV) run pytest -q tests/smoke/test_imports.py

smoke:
	$(UV) run pytest -q tests/smoke/test_imports.py

compose-config:
	docker compose -f $(COMPOSE_FILE) config >/dev/null
	@echo "compose config valid"

run-ml-backend:
	$(UV) run uvicorn services.ml_backend.app.main:app --host 0.0.0.0 --port 9090 --reload

test:
	$(UV) run pytest -q
