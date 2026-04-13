SHELL := /bin/bash

REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PYTHON_BIN ?= $(REPO_ROOT)/.venv/bin/python
PIP_BIN ?= $(REPO_ROOT)/.venv/bin/pip
WORKSPACE_PYTHONPATH := $(REPO_ROOT)/ops/src:$(REPO_ROOT)/shared/birdsys_core/src:$(REPO_ROOT)/projects/labelstudio/src:$(REPO_ROOT)/projects/datasets/src:$(REPO_ROOT)/projects/ml_backend/src:$(REPO_ROOT)/projects/ml_experiments/src

DEPLOY_BRANCH ?= main
PUBLIC_REPO_URL ?= https://github.com/antoinesaget/bird_stance_classification.git

LOCAL_ENV_FILE ?= $(REPO_ROOT)/ops/env/local.env
LOCAL_COMPOSE_FILE ?= $(REPO_ROOT)/ops/compose/docker-compose.local.yml

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
	iats-normalize-annotations iats-build-attributes-dataset iats-evaluate-model-b \
	iats-train-attributes-cv iats-train-attributes-final iats-deploy-model-b \
	truenas-pull truenas-export-annotations truenas-deploy-ui truenas-create-project truenas-prepare-lines-batch truenas-import-lines-batch truenas-prefill-lines-predictions truenas-refresh-lines-predictions \
	smoke-remote

bootstrap:
	python3 -m venv "$(REPO_ROOT)/.venv"
	"$(PIP_BIN)" install -e .

check:
	PYTHONPATH="$(WORKSPACE_PYTHONPATH)" "$(PYTHON_BIN)" -m pytest -q tests/smoke

smoke:
	PYTHONPATH="$(WORKSPACE_PYTHONPATH)" "$(PYTHON_BIN)" -m pytest -q tests/smoke

test:
	PYTHONPATH="$(WORKSPACE_PYTHONPATH)" "$(PYTHON_BIN)" -m pytest -q

local-config:
	ENV_FILE="$(LOCAL_ENV_FILE)" "$(REPO_ROOT)/ops/local_compose.sh" config >/dev/null
	@echo "local compose config valid"

local-up:
	ENV_FILE="$(LOCAL_ENV_FILE)" "$(REPO_ROOT)/ops/local_compose.sh" up

local-down:
	ENV_FILE="$(LOCAL_ENV_FILE)" "$(REPO_ROOT)/ops/local_compose.sh" down

local-ps:
	ENV_FILE="$(LOCAL_ENV_FILE)" "$(REPO_ROOT)/ops/local_compose.sh" ps

local-run-ml-host:
	set -a; source "$(LOCAL_ENV_FILE)"; set +a; \
	PYTHONPATH="$(WORKSPACE_PYTHONPATH)" REPO_ROOT="$(REPO_ROOT)" BIRDS_DATA_ROOT="$${BIRDS_DATA_ROOT}" MODEL_A_WEIGHTS="$${MODEL_A_SERVING_WEIGHTS}" \
	"$(PYTHON_BIN)" -m uvicorn birdsys.ml_backend.app.main:app --host 0.0.0.0 --port "$${ML_BACKEND_HOST_PORT:-9091}" --reload

local-stop-ml-container:
	ENV_FILE="$(LOCAL_ENV_FILE)" "$(REPO_ROOT)/ops/local_compose.sh" stop-ml

compose-config: local-config
compose-up: local-up
compose-down: local-down
compose-ps: local-ps
run-ml-backend: local-up
run-ml-backend-host: local-run-ml-host
stop-ml-backend-container: local-stop-ml-container

iats-pull:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(IATS_REMOTE_REPO_URL)" REMOTE_PUSH_URL="$(IATS_REMOTE_PUSH_URL)" \
	"$(REPO_ROOT)/ops/remote_repo_exec.sh" "$(IATS_HOST)" "$(IATS_REPO_ROOT)" ops/git_pull_ff_only.sh

iats-sync-data:
	IATS_HOST="$(IATS_HOST)" IATS_REPO_ROOT="$(IATS_REPO_ROOT)" TRUENAS_HOST="$(TRUENAS_HOST)" \
	"$(REPO_ROOT)/ops/iats_sync_data.sh"

iats-import-exports:
	IATS_HOST="$(IATS_HOST)" IATS_REPO_ROOT="$(IATS_REPO_ROOT)" TRUENAS_HOST="$(TRUENAS_HOST)" \
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(TRUENAS_REMOTE_REPO_URL)" \
	PROJECT_ID="$(PROJECT_ID)" ANNOTATION_VERSION="$(ANNOTATION_VERSION)" EXPORT_NAME="$(EXPORT_NAME)" \
	EXPORT_TITLE="$(EXPORT_TITLE)" DOWNLOAD_ALL_TASKS="$(DOWNLOAD_ALL_TASKS)" NORMALIZE_ON_IATS="$(NORMALIZE_ON_IATS)" \
	"$(REPO_ROOT)/ops/iats_import_exports.sh"

iats-normalize-annotations:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(IATS_REMOTE_REPO_URL)" REMOTE_PUSH_URL="$(IATS_REMOTE_PUSH_URL)" \
	ANNOTATION_VERSION="$(ANNOTATION_VERSION)" EXPORT_NAME="$(EXPORT_NAME)" EXPORT_JSON="$(EXPORT_JSON)" \
	"$(REPO_ROOT)/ops/remote_repo_exec.sh" "$(IATS_HOST)" "$(IATS_REPO_ROOT)" ops/iats_normalize_annotations_remote.sh

iats-build-attributes-dataset:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(IATS_REMOTE_REPO_URL)" REMOTE_PUSH_URL="$(IATS_REMOTE_PUSH_URL)" \
	ANNOTATION_VERSION="$(ANNOTATION_VERSION)" DATASET_VERSION="$(DATASET_VERSION)" TRAIN_PCT="$(TRAIN_PCT)" VAL_PCT="$(VAL_PCT)" TEST_PCT="$(TEST_PCT)" \
	"$(REPO_ROOT)/ops/remote_repo_exec.sh" "$(IATS_HOST)" "$(IATS_REPO_ROOT)" ops/iats_build_attributes_dataset_remote.sh

iats-evaluate-model-b:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(IATS_REMOTE_REPO_URL)" REMOTE_PUSH_URL="$(IATS_REMOTE_PUSH_URL)" \
	DATASET_DIR="$(DATASET_DIR)" DATASET_VERSION="$(DATASET_VERSION)" CHECKPOINT="$(CHECKPOINT)" SPLIT="$(SPLIT)" EVAL_OUTPUT_DIR="$(EVAL_OUTPUT_DIR)" EVAL_ARGS="$(EVAL_ARGS)" \
	"$(REPO_ROOT)/ops/remote_repo_exec.sh" "$(IATS_HOST)" "$(IATS_REPO_ROOT)" ops/iats_evaluate_model_b_remote.sh

iats-train:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(IATS_REMOTE_REPO_URL)" REMOTE_PUSH_URL="$(IATS_REMOTE_PUSH_URL)" \
	TRAIN_PIPELINE="$(TRAIN_PIPELINE)" DATASET_DIR="$(DATASET_DIR)" DATASET_VERSION="$(DATASET_VERSION)" TRAIN_ARGS="$(TRAIN_ARGS)" TRAIN_SMOKE="$(TRAIN_SMOKE)" TRAIN_CMD="$(TRAIN_CMD)" \
	"$(REPO_ROOT)/ops/remote_repo_exec.sh" "$(IATS_HOST)" "$(IATS_REPO_ROOT)" ops/iats_train_remote.sh

iats-promote-model:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(IATS_REMOTE_REPO_URL)" REMOTE_PUSH_URL="$(IATS_REMOTE_PUSH_URL)" \
	PROMOTION_SOURCE="$(PROMOTION_SOURCE)" PROMOTION_LABEL="$(PROMOTION_LABEL)" PROMOTION_NOTES="$(PROMOTION_NOTES)" \
	"$(REPO_ROOT)/ops/remote_repo_exec.sh" "$(IATS_HOST)" "$(IATS_REPO_ROOT)" ops/iats_promote_model_remote.sh

iats-deploy-ml:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(IATS_REMOTE_REPO_URL)" REMOTE_PUSH_URL="$(IATS_REMOTE_PUSH_URL)" \
	"$(REPO_ROOT)/ops/remote_repo_exec.sh" "$(IATS_HOST)" "$(IATS_REPO_ROOT)" ops/iats_deploy_ml_remote.sh

iats-train-attributes-cv:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(IATS_REMOTE_REPO_URL)" REMOTE_PUSH_URL="$(IATS_REMOTE_PUSH_URL)" \
	DATASET_DIR="$(DATASET_DIR)" DATASET_VERSION="$(DATASET_VERSION)" TRAIN_ARGS="$(TRAIN_ARGS)" TRAIN_SMOKE="$(TRAIN_SMOKE)" \
	"$(REPO_ROOT)/ops/remote_repo_exec.sh" "$(IATS_HOST)" "$(IATS_REPO_ROOT)" ops/iats_train_attributes_cv_remote.sh

iats-train-attributes-final:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(IATS_REMOTE_REPO_URL)" REMOTE_PUSH_URL="$(IATS_REMOTE_PUSH_URL)" \
	DATASET_DIR="$(DATASET_DIR)" DATASET_VERSION="$(DATASET_VERSION)" TRAIN_ARGS="$(TRAIN_ARGS)" TRAIN_SMOKE="$(TRAIN_SMOKE)" TRAIN_SPLIT="$(TRAIN_SPLIT)" TRAIN_EVAL_SPLIT="$(TRAIN_EVAL_SPLIT)" \
	"$(REPO_ROOT)/ops/remote_repo_exec.sh" "$(IATS_HOST)" "$(IATS_REPO_ROOT)" ops/iats_train_attributes_final_remote.sh

iats-deploy-model-b:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(IATS_REMOTE_REPO_URL)" REMOTE_PUSH_URL="$(IATS_REMOTE_PUSH_URL)" \
	MODEL_B_SOURCE="$(MODEL_B_SOURCE)" PROMOTION_LABEL="$(PROMOTION_LABEL)" PROMOTION_NOTES="$(PROMOTION_NOTES)" \
	"$(REPO_ROOT)/ops/remote_repo_exec.sh" "$(IATS_HOST)" "$(IATS_REPO_ROOT)" ops/iats_deploy_model_b_remote.sh

truenas-pull:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(TRUENAS_REMOTE_REPO_URL)" \
	"$(REPO_ROOT)/ops/remote_repo_exec.sh" "$(TRUENAS_HOST)" "$(TRUENAS_REPO_ROOT)" ops/git_pull_ff_only.sh

truenas-export-annotations:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(TRUENAS_REMOTE_REPO_URL)" \
	PROJECT_ID="$(PROJECT_ID)" ANNOTATION_VERSION="$(ANNOTATION_VERSION)" EXPORT_NAME="$(EXPORT_NAME)" \
	EXPORT_TITLE="$(EXPORT_TITLE)" DOWNLOAD_ALL_TASKS="$(DOWNLOAD_ALL_TASKS)" \
	"$(REPO_ROOT)/ops/remote_repo_exec.sh" "$(TRUENAS_HOST)" "$(TRUENAS_REPO_ROOT)" ops/truenas_export_annotations_remote.sh

truenas-deploy-ui:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(TRUENAS_REMOTE_REPO_URL)" \
	"$(REPO_ROOT)/ops/remote_repo_exec.sh" "$(TRUENAS_HOST)" "$(TRUENAS_REPO_ROOT)" ops/truenas_deploy_ui_remote.sh

truenas-create-project:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(TRUENAS_REMOTE_REPO_URL)" \
	SOURCE_PROJECT_ID="$(SOURCE_PROJECT_ID)" TARGET_PROJECT_TITLE="$(TARGET_PROJECT_TITLE)" \
	LABEL_STUDIO_ML_BACKEND_URL="$(LABEL_STUDIO_ML_BACKEND_URL)" LABEL_STUDIO_ML_TITLE="$(LABEL_STUDIO_ML_TITLE)" \
	"$(REPO_ROOT)/ops/remote_repo_exec.sh" "$(TRUENAS_HOST)" "$(TRUENAS_REPO_ROOT)" ops/truenas_create_project_remote.sh

truenas-prepare-lines-batch:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(TRUENAS_REMOTE_REPO_URL)" \
	LINES_DATA_ROOT="$(LINES_DATA_ROOT)" LINES_SOURCE_RELATIVE_ROOT="$(LINES_SOURCE_RELATIVE_ROOT)" \
	LINES_IMPORT_RELATIVE_ROOT="$(LINES_IMPORT_RELATIVE_ROOT)" LINES_SAMPLE_SIZE="$(LINES_SAMPLE_SIZE)" \
	LINES_SAMPLE_MODE="$(LINES_SAMPLE_MODE)" LINES_SAMPLE_SEED="$(LINES_SAMPLE_SEED)" \
	LINES_JPEG_QUALITY="$(LINES_JPEG_QUALITY)" LINES_MIRROR_RELATIVE_ROOT="$(LINES_MIRROR_RELATIVE_ROOT)" \
	LINES_DATASET_NAME="$(LINES_DATASET_NAME)" LINES_BATCH_NAME="$(LINES_BATCH_NAME)" \
	LINES_RECURSIVE="$(LINES_RECURSIVE)" OVERWRITE="$(OVERWRITE)" \
	"$(REPO_ROOT)/ops/remote_repo_exec.sh" "$(TRUENAS_HOST)" "$(TRUENAS_REPO_ROOT)" ops/truenas_prepare_lines_batch_remote.sh

truenas-import-lines-batch:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(TRUENAS_REMOTE_REPO_URL)" \
	LINES_DATA_ROOT="$(LINES_DATA_ROOT)" LINES_IMPORT_RELATIVE_ROOT="$(LINES_IMPORT_RELATIVE_ROOT)" \
	LINES_SAMPLE_SIZE="$(LINES_SAMPLE_SIZE)" LINES_SAMPLE_SEED="$(LINES_SAMPLE_SEED)" \
	LINES_JPEG_QUALITY="$(LINES_JPEG_QUALITY)" LINES_BATCH_NAME="$(LINES_BATCH_NAME)" \
	LINES_PROJECT_ID="$(LINES_PROJECT_ID)" LINES_TASKS_JSON="$(LINES_TASKS_JSON)" \
	LINES_IMPORT_REPORT_OUT="$(LINES_IMPORT_REPORT_OUT)" \
	"$(REPO_ROOT)/ops/remote_repo_exec.sh" "$(TRUENAS_HOST)" "$(TRUENAS_REPO_ROOT)" ops/truenas_import_lines_batch_remote.sh

truenas-prefill-lines-predictions:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(TRUENAS_REMOTE_REPO_URL)" \
	LINES_DATA_ROOT="$(LINES_DATA_ROOT)" LINES_IMPORT_RELATIVE_ROOT="$(LINES_IMPORT_RELATIVE_ROOT)" \
	LINES_SAMPLE_SIZE="$(LINES_SAMPLE_SIZE)" LINES_SAMPLE_SEED="$(LINES_SAMPLE_SEED)" \
	LINES_JPEG_QUALITY="$(LINES_JPEG_QUALITY)" LINES_BATCH_NAME="$(LINES_BATCH_NAME)" \
	LINES_PROJECT_ID="$(LINES_PROJECT_ID)" LINES_ML_BACKEND_URL="$(LINES_ML_BACKEND_URL)" \
	LINES_TASK_PAGE_SIZE="$(LINES_TASK_PAGE_SIZE)" LINES_PREDICT_BATCH_SIZE="$(LINES_PREDICT_BATCH_SIZE)" \
	LINES_PREDICTION_IMPORT_BATCH_SIZE="$(LINES_PREDICTION_IMPORT_BATCH_SIZE)" LINES_ONLY_MISSING="$(LINES_ONLY_MISSING)" \
	LINES_LIMIT="$(LINES_LIMIT)" LINES_PREDICTION_REPORT_OUT="$(LINES_PREDICTION_REPORT_OUT)" \
	"$(REPO_ROOT)/ops/remote_repo_exec.sh" "$(TRUENAS_HOST)" "$(TRUENAS_REPO_ROOT)" ops/truenas_prefill_lines_predictions_remote.sh

truenas-refresh-lines-predictions:
	DEPLOY_BRANCH="$(DEPLOY_BRANCH)" REMOTE_REPO_URL="$(TRUENAS_REMOTE_REPO_URL)" \
	LINES_DATA_ROOT="$(LINES_DATA_ROOT)" LINES_IMPORT_RELATIVE_ROOT="$(LINES_IMPORT_RELATIVE_ROOT)" \
	LINES_SAMPLE_SIZE="$(LINES_SAMPLE_SIZE)" LINES_SAMPLE_SEED="$(LINES_SAMPLE_SEED)" \
	LINES_JPEG_QUALITY="$(LINES_JPEG_QUALITY)" LINES_BATCH_NAME="$(LINES_BATCH_NAME)" \
	LINES_PROJECT_ID="$(LINES_PROJECT_ID)" LINES_ML_BACKEND_URL="$(LINES_ML_BACKEND_URL)" \
	LINES_TASK_PAGE_SIZE="$(LINES_TASK_PAGE_SIZE)" LINES_PREDICT_BATCH_SIZE="$(LINES_PREDICT_BATCH_SIZE)" \
	LINES_PREDICTION_IMPORT_BATCH_SIZE="$(LINES_PREDICTION_IMPORT_BATCH_SIZE)" \
	LINES_LIMIT="$(LINES_LIMIT)" LINES_UNTOUCHED_ONLY="$(LINES_UNTOUCHED_ONLY)" \
	LINES_REPLACE_EXISTING="$(LINES_REPLACE_EXISTING)" LINES_PREDICTION_REPORT_OUT="$(LINES_PREDICTION_REPORT_OUT)" \
	"$(REPO_ROOT)/ops/remote_repo_exec.sh" "$(TRUENAS_HOST)" "$(TRUENAS_REPO_ROOT)" ops/truenas_refresh_lines_predictions_remote.sh

smoke-remote:
	IATS_HOST="$(IATS_HOST)" TRUENAS_HOST="$(TRUENAS_HOST)" \
	"$(REPO_ROOT)/ops/smoke_remote.sh"
