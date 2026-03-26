SHELL := /bin/bash

REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

UV ?= uv
PYTHON_VERSION ?= 3.11
DEPLOY_BRANCH ?= main
PUBLIC_REPO_URL ?= https://github.com/antoinesaget/bird_stance_classification.git

LOCAL_ENV_FILE ?= $(REPO_ROOT)/deploy/env/local.env
LOCAL_COMPOSE_FILE ?= $(REPO_ROOT)/deploy/docker-compose.local.yml

IATS_HOST ?= iats
IATS_REPO_ROOT ?= /home/antoine/bird_stance_classification
IATS_REMOTE_REPO_URL ?= $(PUBLIC_REPO_URL)
IATS_REMOTE_PUSH_URL ?= git@github.com:antoinesaget/bird_stance_classification.git

TRUENAS_HOST ?= truenas
TRUENAS_REPO_ROOT ?= /mnt/apps/code/bird_stance_classification
TRUENAS_REMOTE_REPO_URL ?= $(PUBLIC_REPO_URL)

.PHONY: \
	bootstrap check test smoke \
	local-config local-up local-down local-ps local-run-ml-host local-stop-ml-container \
	compose-config compose-up compose-down compose-ps run-ml-backend run-ml-backend-host stop-ml-backend-container \
	iats-pull iats-sync-data iats-import-exports iats-train iats-promote-model iats-deploy-ml \
	iats-normalize-annotations iats-build-attributes-dataset iats-train-attributes-cv iats-train-attributes-final iats-deploy-model-b \
	truenas-pull truenas-export-annotations truenas-deploy-ui truenas-create-project truenas-prepare-lines-batch truenas-import-lines-batch truenas-prefill-lines-predictions \
	smoke-remote labelstudio-bootstrap-users

bootstrap:
	$(UV) sync --python $(PYTHON_VERSION)

check:
	$(UV) run python -V
	$(UV) run pytest -q tests/smoke

smoke:
	$(UV) run pytest -q tests/smoke

test:
	$(UV) run pytest -q

local-config:
	ENV_FILE="$(LOCAL_ENV_FILE)" "$(REPO_ROOT)/scripts/ops/local_compose.sh" config >/dev/null
	@echo "local compose config valid"

local-up:
	ENV_FILE="$(LOCAL_ENV_FILE)" "$(REPO_ROOT)/scripts/ops/local_compose.sh" up

local-down:
	ENV_FILE="$(LOCAL_ENV_FILE)" "$(REPO_ROOT)/scripts/ops/local_compose.sh" down

local-ps:
	ENV_FILE="$(LOCAL_ENV_FILE)" "$(REPO_ROOT)/scripts/ops/local_compose.sh" ps

local-run-ml-host:
	set -a; source "$(LOCAL_ENV_FILE)"; set +a; \
	REPO_ROOT="$(REPO_ROOT)" BIRDS_DATA_ROOT="$${BIRDS_DATA_ROOT}" MODEL_A_WEIGHTS="$${MODEL_A_SERVING_WEIGHTS}" \
	$(UV) run uvicorn services.ml_backend.app.main:app --host 0.0.0.0 --port "$${ML_BACKEND_HOST_PORT:-9091}" --reload

local-stop-ml-container:
	ENV_FILE="$(LOCAL_ENV_FILE)" "$(REPO_ROOT)/scripts/ops/local_compose.sh" stop-ml

compose-config: local-config
compose-up: local-up
compose-down: local-down
compose-ps: local-ps
run-ml-backend: local-up
run-ml-backend-host: local-run-ml-host
stop-ml-backend-container: local-stop-ml-container

iats-pull:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(IATS_REMOTE_REPO_URL)" REMOTE_PUSH_URL="$(IATS_REMOTE_PUSH_URL)" \
	"$(REPO_ROOT)/scripts/ops/remote_repo_exec.sh" "$(IATS_HOST)" "$(IATS_REPO_ROOT)" scripts/ops/git_pull_ff_only.sh

iats-sync-data:
	IATS_HOST="$(IATS_HOST)" IATS_REPO_ROOT="$(IATS_REPO_ROOT)" TRUENAS_HOST="$(TRUENAS_HOST)" \
	"$(REPO_ROOT)/scripts/ops/iats_sync_data.sh"

iats-import-exports:
	IATS_HOST="$(IATS_HOST)" IATS_REPO_ROOT="$(IATS_REPO_ROOT)" TRUENAS_HOST="$(TRUENAS_HOST)" \
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(TRUENAS_REMOTE_REPO_URL)" \
	PROJECT_ID="$(PROJECT_ID)" ANNOTATION_VERSION="$(ANNOTATION_VERSION)" EXPORT_NAME="$(EXPORT_NAME)" \
	EXPORT_TITLE="$(EXPORT_TITLE)" DOWNLOAD_ALL_TASKS="$(DOWNLOAD_ALL_TASKS)" NORMALIZE_ON_IATS="$(NORMALIZE_ON_IATS)" \
	"$(REPO_ROOT)/scripts/ops/iats_import_exports.sh"

iats-normalize-annotations:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(IATS_REMOTE_REPO_URL)" REMOTE_PUSH_URL="$(IATS_REMOTE_PUSH_URL)" \
	ANNOTATION_VERSION="$(ANNOTATION_VERSION)" EXPORT_NAME="$(EXPORT_NAME)" EXPORT_JSON="$(EXPORT_JSON)" \
	"$(REPO_ROOT)/scripts/ops/remote_repo_exec.sh" "$(IATS_HOST)" "$(IATS_REPO_ROOT)" scripts/ops/iats_normalize_annotations_remote.sh

iats-build-attributes-dataset:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(IATS_REMOTE_REPO_URL)" REMOTE_PUSH_URL="$(IATS_REMOTE_PUSH_URL)" \
	ANNOTATION_VERSION="$(ANNOTATION_VERSION)" DATASET_VERSION="$(DATASET_VERSION)" TRAIN_PCT="$(TRAIN_PCT)" VAL_PCT="$(VAL_PCT)" TEST_PCT="$(TEST_PCT)" \
	"$(REPO_ROOT)/scripts/ops/remote_repo_exec.sh" "$(IATS_HOST)" "$(IATS_REPO_ROOT)" scripts/ops/iats_build_attributes_dataset_remote.sh

iats-train:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(IATS_REMOTE_REPO_URL)" REMOTE_PUSH_URL="$(IATS_REMOTE_PUSH_URL)" \
	TRAIN_PIPELINE="$(TRAIN_PIPELINE)" DATASET_DIR="$(DATASET_DIR)" DATASET_VERSION="$(DATASET_VERSION)" \
	ANNOTATION_VERSION="$(ANNOTATION_VERSION)" TRAIN_SMOKE="$(TRAIN_SMOKE)" TRAIN_ARGS="$(TRAIN_ARGS)" TRAIN_CMD="$(TRAIN_CMD)" \
	"$(REPO_ROOT)/scripts/ops/remote_repo_exec.sh" "$(IATS_HOST)" "$(IATS_REPO_ROOT)" scripts/ops/iats_train_remote.sh

iats-promote-model:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(IATS_REMOTE_REPO_URL)" REMOTE_PUSH_URL="$(IATS_REMOTE_PUSH_URL)" \
	PROMOTION_SOURCE="$(PROMOTION_SOURCE)" PROMOTION_LABEL="$(PROMOTION_LABEL)" PROMOTION_NOTES="$(PROMOTION_NOTES)" \
	"$(REPO_ROOT)/scripts/ops/remote_repo_exec.sh" "$(IATS_HOST)" "$(IATS_REPO_ROOT)" scripts/ops/iats_promote_model_remote.sh

iats-deploy-ml:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(IATS_REMOTE_REPO_URL)" REMOTE_PUSH_URL="$(IATS_REMOTE_PUSH_URL)" \
	IATS_STOP_LEGACY_UI="$(IATS_STOP_LEGACY_UI)" REQUIRE_NON_CPU_DEVICE="$(REQUIRE_NON_CPU_DEVICE)" \
	"$(REPO_ROOT)/scripts/ops/remote_repo_exec.sh" "$(IATS_HOST)" "$(IATS_REPO_ROOT)" scripts/ops/iats_deploy_ml_remote.sh

iats-train-attributes-cv:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(IATS_REMOTE_REPO_URL)" REMOTE_PUSH_URL="$(IATS_REMOTE_PUSH_URL)" \
	DATASET_DIR="$(DATASET_DIR)" DATASET_VERSION="$(DATASET_VERSION)" TRAIN_SMOKE="$(TRAIN_SMOKE)" TRAIN_ARGS="$(TRAIN_ARGS)" \
	"$(REPO_ROOT)/scripts/ops/remote_repo_exec.sh" "$(IATS_HOST)" "$(IATS_REPO_ROOT)" scripts/ops/iats_train_attributes_cv_remote.sh

iats-train-attributes-final:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(IATS_REMOTE_REPO_URL)" REMOTE_PUSH_URL="$(IATS_REMOTE_PUSH_URL)" \
	DATASET_DIR="$(DATASET_DIR)" DATASET_VERSION="$(DATASET_VERSION)" TRAIN_SMOKE="$(TRAIN_SMOKE)" TRAIN_ARGS="$(TRAIN_ARGS)" \
	"$(REPO_ROOT)/scripts/ops/remote_repo_exec.sh" "$(IATS_HOST)" "$(IATS_REPO_ROOT)" scripts/ops/iats_train_attributes_final_remote.sh

iats-deploy-model-b:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(IATS_REMOTE_REPO_URL)" REMOTE_PUSH_URL="$(IATS_REMOTE_PUSH_URL)" \
	MODEL_B_SOURCE="$(MODEL_B_SOURCE)" PROMOTION_LABEL="$(PROMOTION_LABEL)" PROMOTION_NOTES="$(PROMOTION_NOTES)" \
	IATS_STOP_LEGACY_UI="$(IATS_STOP_LEGACY_UI)" REQUIRE_NON_CPU_DEVICE="$(REQUIRE_NON_CPU_DEVICE)" \
	"$(REPO_ROOT)/scripts/ops/remote_repo_exec.sh" "$(IATS_HOST)" "$(IATS_REPO_ROOT)" scripts/ops/iats_deploy_model_b_remote.sh

truenas-pull:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(TRUENAS_REMOTE_REPO_URL)" \
	"$(REPO_ROOT)/scripts/ops/remote_repo_exec.sh" "$(TRUENAS_HOST)" "$(TRUENAS_REPO_ROOT)" scripts/ops/git_pull_ff_only.sh

truenas-export-annotations:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(TRUENAS_REMOTE_REPO_URL)" \
	PROJECT_ID="$(PROJECT_ID)" ANNOTATION_VERSION="$(ANNOTATION_VERSION)" EXPORT_NAME="$(EXPORT_NAME)" \
	EXPORT_TITLE="$(EXPORT_TITLE)" DOWNLOAD_ALL_TASKS="$(DOWNLOAD_ALL_TASKS)" \
	"$(REPO_ROOT)/scripts/ops/remote_repo_exec.sh" "$(TRUENAS_HOST)" "$(TRUENAS_REPO_ROOT)" scripts/ops/truenas_export_annotations_remote.sh

truenas-deploy-ui:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(TRUENAS_REMOTE_REPO_URL)" \
	"$(REPO_ROOT)/scripts/ops/remote_repo_exec.sh" "$(TRUENAS_HOST)" "$(TRUENAS_REPO_ROOT)" scripts/ops/truenas_deploy_ui_remote.sh

truenas-create-project:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(TRUENAS_REMOTE_REPO_URL)" \
	SOURCE_PROJECT_ID="$(SOURCE_PROJECT_ID)" TARGET_PROJECT_TITLE="$(TARGET_PROJECT_TITLE)" \
	LABEL_STUDIO_ML_BACKEND_URL="$(LABEL_STUDIO_ML_BACKEND_URL)" LABEL_STUDIO_ML_TITLE="$(LABEL_STUDIO_ML_TITLE)" \
	"$(REPO_ROOT)/scripts/ops/remote_repo_exec.sh" "$(TRUENAS_HOST)" "$(TRUENAS_REPO_ROOT)" scripts/ops/truenas_create_project_remote.sh

truenas-prepare-lines-batch:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(TRUENAS_REMOTE_REPO_URL)" \
	LINES_DATA_ROOT="$(LINES_DATA_ROOT)" LINES_SOURCE_RELATIVE_ROOT="$(LINES_SOURCE_RELATIVE_ROOT)" \
	LINES_IMPORT_RELATIVE_ROOT="$(LINES_IMPORT_RELATIVE_ROOT)" LINES_SAMPLE_SIZE="$(LINES_SAMPLE_SIZE)" \
	LINES_SAMPLE_MODE="$(LINES_SAMPLE_MODE)" LINES_SAMPLE_SEED="$(LINES_SAMPLE_SEED)" \
	LINES_JPEG_QUALITY="$(LINES_JPEG_QUALITY)" LINES_MIRROR_RELATIVE_ROOT="$(LINES_MIRROR_RELATIVE_ROOT)" \
	LINES_DATASET_NAME="$(LINES_DATASET_NAME)" LINES_BATCH_NAME="$(LINES_BATCH_NAME)" \
	LINES_RECURSIVE="$(LINES_RECURSIVE)" OVERWRITE="$(OVERWRITE)" \
	"$(REPO_ROOT)/scripts/ops/remote_repo_exec.sh" "$(TRUENAS_HOST)" "$(TRUENAS_REPO_ROOT)" scripts/ops/truenas_prepare_lines_batch_remote.sh

truenas-import-lines-batch:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(TRUENAS_REMOTE_REPO_URL)" \
	LINES_DATA_ROOT="$(LINES_DATA_ROOT)" LINES_IMPORT_RELATIVE_ROOT="$(LINES_IMPORT_RELATIVE_ROOT)" \
	LINES_SAMPLE_SIZE="$(LINES_SAMPLE_SIZE)" LINES_SAMPLE_SEED="$(LINES_SAMPLE_SEED)" \
	LINES_JPEG_QUALITY="$(LINES_JPEG_QUALITY)" LINES_BATCH_NAME="$(LINES_BATCH_NAME)" \
	LINES_PROJECT_ID="$(LINES_PROJECT_ID)" LINES_TASKS_JSON="$(LINES_TASKS_JSON)" \
	LINES_IMPORT_REPORT_OUT="$(LINES_IMPORT_REPORT_OUT)" \
	"$(REPO_ROOT)/scripts/ops/remote_repo_exec.sh" "$(TRUENAS_HOST)" "$(TRUENAS_REPO_ROOT)" scripts/ops/truenas_import_lines_batch_remote.sh

truenas-prefill-lines-predictions:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(TRUENAS_REMOTE_REPO_URL)" \
	LINES_DATA_ROOT="$(LINES_DATA_ROOT)" LINES_IMPORT_RELATIVE_ROOT="$(LINES_IMPORT_RELATIVE_ROOT)" \
	LINES_SAMPLE_SIZE="$(LINES_SAMPLE_SIZE)" LINES_SAMPLE_SEED="$(LINES_SAMPLE_SEED)" \
	LINES_JPEG_QUALITY="$(LINES_JPEG_QUALITY)" LINES_BATCH_NAME="$(LINES_BATCH_NAME)" \
	LINES_PROJECT_ID="$(LINES_PROJECT_ID)" LINES_ML_BACKEND_URL="$(LINES_ML_BACKEND_URL)" \
	LINES_TASK_PAGE_SIZE="$(LINES_TASK_PAGE_SIZE)" LINES_PREDICT_BATCH_SIZE="$(LINES_PREDICT_BATCH_SIZE)" \
	LINES_PREDICTION_IMPORT_BATCH_SIZE="$(LINES_PREDICTION_IMPORT_BATCH_SIZE)" LINES_ONLY_MISSING="$(LINES_ONLY_MISSING)" \
	LINES_LIMIT="$(LINES_LIMIT)" LINES_PREDICTION_REPORT_OUT="$(LINES_PREDICTION_REPORT_OUT)" \
	"$(REPO_ROOT)/scripts/ops/remote_repo_exec.sh" "$(TRUENAS_HOST)" "$(TRUENAS_REPO_ROOT)" scripts/ops/truenas_prefill_lines_predictions_remote.sh

smoke-remote:
	IATS_HOST="$(IATS_HOST)" TRUENAS_HOST="$(TRUENAS_HOST)" \
	"$(REPO_ROOT)/scripts/ops/smoke_remote.sh"

labelstudio-bootstrap-users:
	./scripts/bootstrap_labelstudio_users.sh
