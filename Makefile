#################################################################################
# GLOBALS
#################################################################################

PROJECT_NAME := chest_xray
PYTHON_VERSION := 3.10
PYTHON_INTERPRETER := python

TRAIN_MODULE := src.train

CONFIG_DIR := configs
ALL_CFGS := $(wildcard $(CONFIG_DIR)/*.yaml)
MODEL_CFGS := $(filter-out $(CONFIG_DIR)/mlflow.yaml,$(ALL_CFGS))

CFG ?= configs/cnn.yaml
SEED ?= 42
SEEDS ?= 0

#################################################################################
# ENV / HYGIENE
#################################################################################

.PHONY: requirements clean lint format test create_environment
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
lint:
	ruff format --check
	ruff check
format:
	ruff check --fix
	ruff format
test:
	$(PYTHON_INTERPRETER) -m unittest discover -s tests
create_environment:
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	@echo "Activate with: conda activate $(PROJECT_NAME)"

#################################################################################
# 1. Model TRAINING - Models (CNN, Backbones+LinearProbe, HOG + MLP/XGBoost/RF)
#################################################################################
.PHONY: train train_one train_hog  sweep_seeds probe_one finetune_one probe_sweep finetune_sweep train_all
train:
	$(PYTHON_INTERPRETER) -m $(TRAIN_MODULE) --config $(CFG) --seed $(SEED)

train_one: train

sweep_seeds:
	@set -e; \
	for s in $(SEEDS); do \
		echo "=== Training $(CFG) seed=$$s ==="; \
		$(PYTHON_INTERPRETER) -m $(TRAIN_MODULE) --config $(CFG) --seed $$s; \
	done

probe_one:
	$(PYTHON_INTERPRETER) -m $(TRAIN_MODULE) --config $(CFG) --linear_probe 1 --seed $(SEED)

finetune_one:
	$(PYTHON_INTERPRETER) -m $(TRAIN_MODULE) --config $(CFG) --linear_probe 0 --seed $(SEED)

probe_sweep:
	@set -e; \
	for s in $(SEEDS); do \
		echo "=== Linear probe $(CFG) seed=$$s ==="; \
		$(PYTHON_INTERPRETER) -m $(TRAIN_MODULE) --config $(CFG) --linear_probe 1 --seed $$s; \
	done

finetune_sweep:
	@set -e; \
	for s in $(SEEDS); do \
		echo "=== Finetune $(CFG) seed=$$s ==="; \
		$(PYTHON_INTERPRETER) -m $(TRAIN_MODULE) --config $(CFG) --linear_probe 0 --seed $$s; \
	done

train_all:
	@set -e; \
	for cfg in $(MODEL_CFGS); do \
		for s in $(SEEDS); do \
			echo "=== Training $$cfg seed=$$s ==="; \
			$(PYTHON_INTERPRETER) -m $(TRAIN_MODULE) --config $$cfg --seed $$s; \
		done; \
	done

# HOG BASELINES
train_hog:
	$(PYTHON_INTERPRETER) -m src.train_hog \
	  --train_csv data/train.csv --val_csv data/val.csv --test_csv data/test.csv \
	  --img_root null --img_col image_path --label_col label \
	  --class_names "Normal,Pneumonia,Tuberculosis" \
	  --out_dir artifacts/hog_baselines --seed 0 \
	  --clahe_clip 2.0 --clahe_grid 8,8

#################################################################################
# 2. Evaluation and Inference 
#################################################################################
.PHONY: eval-best analyze-best analyze-best-only infer infer-image infer-image-demo

OUT_ROOT := reports/best_model_b01
TEST_CSV := data/test.csv
CLASS_NAMES := Normal,Pneumonia,Tuberculosis
BORDERLINE_MARGIN := 0.05
BEST_META := $(OUT_ROOT)/best_model_meta.json

# 1) Select best model --> evaluate --> save meta
eval-best:
	$(PYTHON_INTERPRETER) -m src.eval_best_models --test $(TEST_CSV) --out_root $(OUT_ROOT)

# 2) Error analysis (creates analysis/viz_cases/*.json)
analyze-best: eval-best
	@BEST_RUN_DIR=$$($(PYTHON_INTERPRETER) -m src.utils.read_best_model_meta --meta "$(BEST_META)" --field run_dir); \
	POLICY_FILE="$$BEST_RUN_DIR/thresholds_policy.json"; \
	$(PYTHON_INTERPRETER) -m src.cnn_error_analysis --run_dir "$$BEST_RUN_DIR" \
		--names "$(CLASS_NAMES)" \
		--policy_thresholds "$$POLICY_FILE" \
		--borderline_margin $(BORDERLINE_MARGIN)

analyze-best-only: 
	@BEST_RUN_DIR=$$($(PYTHON_INTERPRETER) -m src.utils.read_best_model_meta --meta "$(BEST_META)" --field run_dir); \
	POLICY_FILE="$$BEST_RUN_DIR/thresholds_policy.json"; \
	$(PYTHON_INTERPRETER) -m src.cnn_error_analysis --run_dir "$$BEST_RUN_DIR" \
		--names "$(CLASS_NAMES)" \
		--policy_thresholds "$$POLICY_FILE" \
		--borderline_margin $(BORDERLINE_MARGIN)

#################################################################################
# 3. Grad-CAM Analysis
#################################################################################
.PHONY: gradcam-tb-fn-true gradcam-tb-fn-pred \
        gradcam-tb-fp-pred \
        gradcam-tb-borderline-fixed gradcam-tb-borderline-topk2 \
        gradcam-correct-normal gradcam-correct-pna gradcam-correct-tb
		
CNN_CFG := configs/cnn.yaml
#OUT_ROOT := reports/best_model_eval
#TEST_CSV := data/test.csv
#BEST_META := $(OUT_ROOT)/best_model_meta.json
#CLASS_NAMES := Normal,Pneumonia,Tuberculosis
MAX_CASES := 20

# => helper fr run dir + run id 
define LOAD_BEST
BEST_RUN_DIR=$$($(PYTHON_INTERPRETER) -m src.utils.read_best_model_meta --meta "$(BEST_META)" --field run_dir); \
BEST_RUN_ID=$$($(PYTHON_INTERPRETER) -m src.utils.read_best_model_meta --meta "$(BEST_META)" --field mlflow_run_id);
endef

#  TB false negatives (TB -> Normal) 
gradcam-tb-fn-true:
	@$(LOAD_BEST) \
	CASE_LIST="$$BEST_RUN_DIR/analysis/viz_cases/tb_fn__tb_to_normal.json"; \
	$(PYTHON_INTERPRETER) -m src.run_gradcam --test_csv $(TEST_CSV) --run_id "$$BEST_RUN_ID" \
	  --cnn_cfg $(CNN_CFG) --out_dir "$$BEST_RUN_DIR/gradcam_tb_fn_true" \
	  --case_list_json "$$CASE_LIST" --names "$(CLASS_NAMES)" --max_cases $(MAX_CASES) \
	  --cam_mode true --method smoothgradcampp --smooth_n 25 --smooth_std 0.10

gradcam-tb-fn-pred:
	@$(LOAD_BEST) \
	CASE_LIST="$$BEST_RUN_DIR/analysis/viz_cases/tb_fn__tb_to_normal.json"; \
	$(PYTHON_INTERPRETER) -m src.run_gradcam --test_csv $(TEST_CSV) --run_id "$$BEST_RUN_ID" \
	  --cnn_cfg $(CNN_CFG)  --out_dir "$$BEST_RUN_DIR/gradcam_tb_fn_pred" \
	  --case_list_json "$$CASE_LIST" --names "$(CLASS_NAMES)" --max_cases $(MAX_CASES) \
	  --cam_mode pred --method smoothgradcampp --smooth_n 25 --smooth_std 0.10

#  TB false positives (Normal -> TB) 
gradcam-tb-fp-pred:
	@$(LOAD_BEST) \
	CASE_LIST="$$BEST_RUN_DIR/analysis/viz_cases/tb_fp__normal_to_tb.json"; \
	$(PYTHON_INTERPRETER) -m src.run_gradcam --test_csv $(TEST_CSV) --run_id "$$BEST_RUN_ID" \
	  --cnn_cfg $(CNN_CFG)  --out_dir "$$BEST_RUN_DIR/gradcam_tb_fp_pred" \
	  --case_list_json "$$CASE_LIST" --names "$(CLASS_NAMES)" --max_cases $(MAX_CASES) \
	  --cam_mode pred --method smoothgradcampp --smooth_n 25 --smooth_std 0.10

#  TB borderline 
gradcam-tb-borderline-fixed:
	@$(LOAD_BEST) \
	CASE_LIST="$$BEST_RUN_DIR/analysis/viz_cases/tb_borderline.json"; \
	$(PYTHON_INTERPRETER) -m src.run_gradcam --test_csv $(TEST_CSV) --run_id "$$BEST_RUN_ID" \
	  --cnn_cfg $(CNN_CFG)  --out_dir "$$BEST_RUN_DIR/gradcam_tb_borderline_fixed" \
	  --case_list_json "$$CASE_LIST" --names "$(CLASS_NAMES)" --max_cases $(MAX_CASES) \
	  --cam_mode fixed --fixed_class Tuberculosis --method smoothgradcampp --smooth_n 25 --smooth_std 0.10

gradcam-tb-borderline-topk2:
	@$(LOAD_BEST) \
	CASE_LIST="$$BEST_RUN_DIR/analysis/viz_cases/tb_borderline.json"; \
	$(PYTHON_INTERPRETER) -m src.run_gradcam --test_csv $(TEST_CSV) --run_id "$$BEST_RUN_ID" \
	  --cnn_cfg $(CNN_CFG)  --out_dir "$$BEST_RUN_DIR/gradcam_tb_borderline_topk2" \
	  --case_list_json "$$CASE_LIST" --names "$(CLASS_NAMES)" --max_cases $(MAX_CASES) \
	  --cam_mode topk --topk 2 --method gradcampp

#  Correct high-confidence controls 
gradcam-correct-normal:
	@$(LOAD_BEST) \
	CASE_LIST="$$BEST_RUN_DIR/analysis/viz_cases/correct_highconf__normal.json"; \
	$(PYTHON_INTERPRETER) -m src.run_gradcam --test_csv $(TEST_CSV) --run_id "$$BEST_RUN_ID" \
	  --cnn_cfg $(CNN_CFG)  --out_dir "$$BEST_RUN_DIR/gradcam_correct_normal" \
	  --case_list_json "$$CASE_LIST" --names "$(CLASS_NAMES)" --max_cases $(MAX_CASES) \
	  --cam_mode pred --method gradcampp

gradcam-correct-pna:
	@$(LOAD_BEST) \
	CASE_LIST="$$BEST_RUN_DIR/analysis/viz_cases/correct_highconf__pneumonia.json"; \
	$(PYTHON_INTERPRETER) -m src.run_gradcam --test_csv $(TEST_CSV) --run_id "$$BEST_RUN_ID" \
	  --cnn_cfg $(CNN_CFG)  --out_dir "$$BEST_RUN_DIR/gradcam_correct_pna" \
	  --case_list_json "$$CASE_LIST" --names "$(CLASS_NAMES)" --max_cases $(MAX_CASES) \
	  --cam_mode pred --method gradcampp

gradcam-correct-tb:
	@$(LOAD_BEST) \
	CASE_LIST="$$BEST_RUN_DIR/analysis/viz_cases/correct_highconf__tuberculosis.json"; \
	$(PYTHON_INTERPRETER) -m src.run_gradcam --test_csv $(TEST_CSV) --run_id "$$BEST_RUN_ID" \
	  --cnn_cfg $(CNN_CFG) --out_dir "$$BEST_RUN_DIR/gradcam_correct_tb" \
	  --case_list_json "$$CASE_LIST" --names "$(CLASS_NAMES)" --max_cases $(MAX_CASES) \
	  --cam_mode pred --method smoothgradcampp --smooth_n 25 --smooth_std 0.10

#################################################################################
infer:
	@$(LOAD_BEST) \
	POLICY_JSON="$$BEST_RUN_DIR/thresholds_policy.json"; \
	$(PYTHON_INTERPRETER) -m src.inference.infer \
		--run_id "$$BEST_RUN_ID" \
		--csv $(TEST_CSV) --img_col image_path \
		--out reports/inference/$(shell date +%Y%m%d_%H%M%S)_infer.csv \
		--names "$(CLASS_NAMES)" \
		--policy_json "$$POLICY_JSON"

DEMO_IMG := data/raw/archive/test/tuberculosis/tuberculosis-828.jpg
infer-image-demo:
	@$(LOAD_BEST) \
	POLICY_JSON="$$BEST_RUN_DIR/thresholds_policy.json"; \
	$(PYTHON_INTERPRETER) -m src.inference.infer \
		--run_id "$$BEST_RUN_ID" \
		--image "$(DEMO_IMG)" \
		--out reports/inference/$(shell date +%Y%m%d_%H%M%S)_infer.json \
		--names "$(CLASS_NAMES)" \
		--policy_json "$$POLICY_JSON"

infer-image:
	@$(LOAD_BEST) \
	POLICY_JSON="$$BEST_RUN_DIR/thresholds_policy.json"; \
	$(PYTHON_INTERPRETER) -m src.inference.infer \
		--run_id "$$BEST_RUN_ID" \
		--image "$(IMG)" \
		--out reports/inference/infer_one.json \
		--names "$(CLASS_NAMES)" \
		--policy_json "$$POLICY_JSON"
#make infer-image IMG=data/raw/archive/test/tuberculosis/tuberculosis-4.jpg

################################################################################
# HELP
#################################################################################
.DEFAULT_GOAL := help
define PRINT_HELP_PYSCRIPT
import re, sys; \
text = sys.stdin.read(); \
items = re.findall(r'\n## (.*?)\n([a-zA-Z_-]+):', text); \
print('Available commands:\n'); \
for desc, cmd in items: print(f'{cmd:25} {desc}')
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

#################################################################################
# 4. Deployment : MODEL BUNDLE (Artifact from best run)
#################################################################################
.PHONY: bundle-best 
BUNDLES_DIR := bundles
BUNDLE_ID ?= $(shell date +%Y%m%d_%H%M%S)
BUNDLE_DIR := $(BUNDLES_DIR)/chestxray_$(BUNDLE_ID)
SEMVER ?= 0.1.0
bundle-best: analyze-best-only
	@set -e; \
	$(LOAD_BEST) \
	mkdir -p "$(BUNDLE_DIR)"; \
	$(PYTHON_INTERPRETER) -m src.deployment.build_bundle \
	  --mlflow_run_id "$$BEST_RUN_ID" --run_dir "$$BEST_RUN_DIR" --out_dir "$(BUNDLE_DIR)" \
	  --class_names "$(CLASS_NAMES)" --cnn_cfg "$(CFG)" --best_meta "$(BEST_META)" \
	  --semver "$(SEMVER)"; \
	ln -sfn "$(notdir $(BUNDLE_DIR))" "$(BUNDLES_DIR)/latest"; \
	echo "@@ Bundle created at: $(BUNDLE_DIR) (and linked to bundles/latest)"


# DOCKER (Inference with bundle)
.PHONY: prep-bundle-for-docker docker-build-infer docker-assert-latest docker-smoke-baked ci-smoke
# Inputs
BUNDLE ?= bundles/latest

# Build context destination => inside repo for docker build
DOCKER_BUNDLE_DIR := docker/_bundle
# Docker
DOCKER ?= docker
DOCKERFILE_INFER ?= docker/Dockerfile.infer
DOCKER_IMAGE ?= chestxray-infer
DOCKER_TAG ?= dev
DOCKER_REF := $(DOCKER_IMAGE):$(DOCKER_TAG)
# Smoke inputs/outputs
DEMO_IMG ?= data/raw/archive/test/tuberculosis/tuberculosis-828.jpg
SMOKE_OUT ?= reports/inference/docker_smoke.json
prep-bundle-for-docker:
	@set -e; \
	REPO_ROOT="$$(pwd)"; \
	BUNDLE_REAL="$$(python -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$(BUNDLE)")"; \
	echo "REPO_ROOT=$$REPO_ROOT"; \
	echo "BUNDLE=$(BUNDLE)"; \
	echo "BUNDLE_REAL=$$BUNDLE_REAL"; \
	test "$$BUNDLE_REAL" != "/"; \
	echo "$$BUNDLE_REAL" | grep -q "^$$REPO_ROOT/bundles/"; \
	test -f "$$BUNDLE_REAL/manifest.json"; \
	test -d "$$BUNDLE_REAL/model"; \
	rm -rf "$(DOCKER_BUNDLE_DIR)"; \
	mkdir -p "$(DOCKER_BUNDLE_DIR)"; \
	cp -R -X "$$BUNDLE_REAL"/. "$(DOCKER_BUNDLE_DIR)"/; \
	test -f "$(DOCKER_BUNDLE_DIR)/manifest.json"; \
	echo " >>> Copied bundle into $(DOCKER_BUNDLE_DIR)"

docker-build-infer: prep-bundle-for-docker
	$(DOCKER) build -f $(DOCKERFILE_INFER) -t $(DOCKER_REF) .

docker-assert-latest: docker-build-infer
	@set -e; \
	HOST_RUN_ID="$$(python -c 'import json; print(json.load(open("$(DOCKER_BUNDLE_DIR)/manifest.json"))["mlflow_run_id"])')"; \
	IMG_RUN_ID="$$( $(DOCKER) run --rm $(DOCKER_REF) python -c 'import json; print(json.load(open("/bundle/manifest.json"))["mlflow_run_id"])' )"; \
	echo "Host bundle run_id: $$HOST_RUN_ID"; \
	echo "Image bundle run_id: $$IMG_RUN_ID"; \
	test "$$HOST_RUN_ID" = "$$IMG_RUN_ID"; \
	echo ">>> Image contains latest bundle"
docker-smoke-baked: docker-build-infer docker-assert-latest
	@set -e; \
	mkdir -p $(dir $(SMOKE_OUT)); \
	$(DOCKER) run --rm \
	  -v "$(PWD):/work" \
	  $(DOCKER_REF) \
	  python -m src.inference.infer_deployment \
	    --image "/work/$(DEMO_IMG)" \
	    --out "/work/$(SMOKE_OUT)"; \
	test -f "$(SMOKE_OUT)"; \
	python -c 'import json; d=json.load(open("$(SMOKE_OUT)")); assert d.get("pred_policy_id") is not None, "policy not applied"; print("> policy applied:", d.get("pred_policy_name"))'

ci-smoke: docker-smoke-baked
	@echo "CI smoke gate passed"

.PHONY: docker-infer-one
# make docker-infer-one IMG=/Users/lily/Documents/Projects/tuberculosis-9.jpg

docker-infer-one:
	@test -n "$(IMG)" || (echo "Provide IMG=/absolute/path/to/image.jpg" && exit 1)
	@test -f "$(IMG)" || (echo "Image not found: $(IMG)" && exit 1)
	@IMG_ABS="$(IMG)"; \
	IMG_DIR="$$(dirname "$$IMG_ABS")"; \
	IMG_BASE="$$(basename "$$IMG_ABS")"; \
	NAME="$${IMG_BASE%.*}"; \
	OUT="$$IMG_DIR/$${NAME}_pred.json"; \
	echo "Image: $$IMG_ABS"; \
	echo "Output: $$OUT"; \
	docker run --rm \
	  -v "$$IMG_DIR:/input:ro" \
	  -v "$$IMG_DIR:/output" \
	  chestxray-infer:dev \
	  python -m src.inference.infer_deployment \
	    --image "/input/$$IMG_BASE" \
	    --out "/output/$${NAME}_pred.json"; \
	test -f "$$OUT"; \
	echo "Saved prediction: $$OUT"

.PHONY: docker-infer-folder 

INPUT_DIR ?= $(HOME)/Documents/Projects/test-data
OUTPUT_DIR ?= $(HOME)/Documents/Projects/test-data-result
BATCH_CSV ?= $(OUTPUT_DIR)/batch.csv

docker-infer-folder: docker-build-infer docker-assert-latest
	@mkdir -p "$(OUTPUT_DIR)"
	python src/utils/make_batch_csv.py --input_dir "$(INPUT_DIR)" --out_csv "$(BATCH_CSV)"
	@echo "First 5 rows of batch.csv:"; head -n 6 "$(BATCH_CSV)"
	docker run --rm \
	  -v "$(INPUT_DIR):/input:ro" \
	  -v "$(OUTPUT_DIR):/output" \
	  chestxray-infer:dev \
	  python -m src.inference.infer_deployment \
	    --csv /output/batch.csv \
	    --img_col image_path \
	    --out /output/predictions.csv
	@echo " Saved: $(OUTPUT_DIR)/predictions.csv"


#################################################################################
# 5. Deployment : DOCKER API (FastAPI)
#################################################################################

.PHONY: docker-build-api docker-run-api docker-api-health docker-api-info docker-api-metrics \
        docker-api-smoke ci-api docker-api-save docker-api-load docker-api-run-remote

# Docker
DOCKER ?= docker
DOCKERFILE_API ?= docker/Dockerfile.api
DOCKER_API_IMAGE ?= chestxray-api
DOCKER_API_TAG ?= dev
DOCKER_API_REF := $(DOCKER_API_IMAGE):$(DOCKER_API_TAG)

# Runtime
API_PORT ?= 8000
API_TOKEN ?= chestxray

API_BASE := http://localhost:$(API_PORT)

# Smoke inputs/outputs
DEMO_IMG ?= data/raw/archive/test/tuberculosis/tuberculosis-828.jpg
TMPDIR ?= /tmp
SMOKE_JSON := $(TMPDIR)/cxr_pred.json

# Export
EXPORT_DIR ?= dist
EXPORT_NAME ?= $(DOCKER_API_IMAGE)_$(DOCKER_API_TAG).tar.gz
EXPORT_PATH := $(EXPORT_DIR)/$(EXPORT_NAME)

define CURL_AUTH
if [ -n "$(API_TOKEN)" ]; then \
  echo "-H Authorization: Bearer $(API_TOKEN)"; \
fi
endef

docker-build-api: prep-bundle-for-docker
	$(DOCKER) build -f $(DOCKERFILE_API) -t $(DOCKER_API_REF) .

docker-run-api: docker-build-api
	$(DOCKER) run --rm -p $(API_PORT):8000 \
	  -e API_TOKEN="$(API_TOKEN)" \
	  $(DOCKER_API_REF)

docker-api-health:
	curl -fsS "$(API_BASE)/health" | python -m json.tool

docker-api-info:
	curl -fsS "$(API_BASE)/info" | python -m json.tool

docker-api-metrics:
	curl -fsS "$(API_BASE)/metrics" | head

docker-api-smoke: docker-build-api
	@set -e; \
	CID=$$($(DOCKER) run -d -p $(API_PORT):8000 -e API_TOKEN="$(API_TOKEN)" $(DOCKER_API_REF)); \
	echo "Started $$CID"; \
	trap '$(DOCKER) rm -f $$CID >/dev/null 2>&1 || true' EXIT; \
	\
	# Wait up to ~10s for health to come up
	for i in 1 2 3 4 5 6 7 8 9 10; do \
	  if curl -fsS "$(API_BASE)/health" >/dev/null; then break; fi; \
	  sleep 1; \
	done; \
	curl -fsS "$(API_BASE)/health" >/dev/null; \
	echo ">>> /health ok"; \
	\
	AUTHH=""; \
	if [ -n "$(API_TOKEN)" ]; then AUTHH="-H Authorization: Bearer $(API_TOKEN)"; fi; \
	curl -fsS $$AUTHH -X POST "$(API_BASE)/predict" \
	  -F "file=@$(DEMO_IMG)" > "$(SMOKE_JSON)"; \
	python -c 'import json; d=json.load(open("$(SMOKE_JSON)")); assert "pred_name" in d; print("/predict ok:", d.get("pred_policy_name"))'; \
	echo ">>> API smoke passed"

ci-api: docker-api-smoke
	@echo ">>> CI API gate passed"

# Build + export compressed image tarball
docker-api-save: docker-build-api
	@set -e; \
	mkdir -p "$(EXPORT_DIR)"; \
	echo "Saving image $(DOCKER_API_REF) -> $(EXPORT_PATH)"; \
	$(DOCKER) save "$(DOCKER_API_REF)" | gzip > "$(EXPORT_PATH)"; \
	ls -lh "$(EXPORT_PATH)"

# Import the image tarball into docker
# Usage: make docker-api-load EXPORT_PATH=/path/to/chestxray-api_dev.tar.gz
docker-api-load:
	@set -e; \
	test -f "$(EXPORT_PATH)" || (echo "Missing EXPORT_PATH: $(EXPORT_PATH)"; exit 1); \
	echo "Loading image from $(EXPORT_PATH)"; \
	gunzip -c "$(EXPORT_PATH)" | $(DOCKER) load; \
	$(DOCKER) images | head

# Run detached on target machine with restart policy
# Usage: make docker-api-run-remote API_PORT=8000 API_TOKEN=chestxray
docker-api-run-remote:
	@set -e; \
	$(DOCKER) rm -f chestxray-api >/dev/null 2>&1 || true; \
	$(DOCKER) run -d --restart unless-stopped \
	  --name chestxray-api \
	  -p $(API_PORT):8000 \
	  -e API_TOKEN="$(API_TOKEN)" \
	  $(DOCKER_API_REF); \
	echo "Running: chestxray-api on port $(API_PORT)"; \
	$(DOCKER) ps --filter name=chestxray-api
