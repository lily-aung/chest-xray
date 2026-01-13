#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = chest-xray
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

# Training entrypoint
TRAIN_MODULE = -m src.train

# Config discovery
CONFIG_DIR := configs
ALL_CFGS := $(wildcard $(CONFIG_DIR)/*.yaml)
MODEL_CFGS := $(filter-out $(CONFIG_DIR)/mlflow.yaml,$(ALL_CFGS))

# Default backbones list (optional override)
# If you prefer auto-discovery only, you can ignore BACKBONES and use MODEL_CFGS
BACKBONES ?= configs/efficientnet_b0.yaml configs/densenet121.yaml configs/swin_tiny_patch4_window7_224.yaml

# Defaults for training automation
CFG ?= configs/resnet50.yaml
SEED ?= 0
SEEDS ?= 0

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Run tests
.PHONY: test
test:
	python -m unittest discover -s tests

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"

#################################################################################
# TRAINING AUTOMATION                                                           #
#################################################################################

## Train default config (CFG) with default seed (SEED)
.PHONY: train
train:
	$(PYTHON_INTERPRETER) $(TRAIN_MODULE) --config $(CFG) --seed $(SEED)

# Histogram of Gradient : MLP/RF/XGB
train_hog:
	python -m src.train_hog \
	  --train_csv data/train.csv --val_csv data/val.csv --test_csv data/test.csv \
	  --img_root null --img_col image_path --label_col label \
	  --class_names "Normal,Pneumonia,Tuberculosis" \
	  --out_dir artifacts/hog_baselines --seed 0 \
	  --clahe_clip 2.0 --clahe_grid 8,8


## Train one config: make train_one CFG=configs/resnet50.yaml SEED=0
.PHONY: train_one
train_one:
	$(PYTHON_INTERPRETER) $(TRAIN_MODULE) --config $(CFG) --seed $(SEED)

## Sweep seeds for one config: make sweep_seeds CFG=... SEEDS="0 1 2 3 4"
.PHONY: sweep_seeds
sweep_seeds:
	@set -e; \
	for s in $(SEEDS); do \
		echo "=== Training $(CFG) seed=$$s ==="; \
		$(PYTHON_INTERPRETER) $(TRAIN_MODULE) --config $(CFG) --seed $$s; \
	done

## Linear probe one config: make probe_one CFG=... SEED=0
.PHONY: probe_one
probe_one:
	$(PYTHON_INTERPRETER) $(TRAIN_MODULE) --config $(CFG) --linear_probe 1 --seed $(SEED)

## Finetune one config: make finetune_one CFG=... SEED=0
.PHONY: finetune_one
finetune_one:
	$(PYTHON_INTERPRETER) $(TRAIN_MODULE) --config $(CFG) --linear_probe 0 --seed $(SEED)

## Linear probe sweep for one config: make probe_sweep CFG=... SEEDS="0 1 2"
.PHONY: probe_sweep
probe_sweep:
	@set -e; \
	for s in $(SEEDS); do \
		echo "=== Linear probe $(CFG) seed=$$s ==="; \
		$(PYTHON_INTERPRETER) $(TRAIN_MODULE) --config $(CFG) --linear_probe 1 --seed $$s; \
	done

## Finetune sweep for one config: make finetune_sweep CFG=... SEEDS="0 1 2"
.PHONY: finetune_sweep
finetune_sweep:
	@set -e; \
	for s in $(SEEDS); do \
		echo "=== Finetune $(CFG) seed=$$s ==="; \
		$(PYTHON_INTERPRETER) $(TRAIN_MODULE) --config $(CFG) --linear_probe 0 --seed $$s; \
	done

## Train all discovered model configs (excluding mlflow.yaml), with SEEDS list
## Example: make train_all SEEDS="0 1 2"
.PHONY: train_all
train_all:
	@set -e; \
	for cfg in $(MODEL_CFGS); do \
		for s in $(SEEDS); do \
			echo "=== Training $$cfg seed=$$s ==="; \
			$(PYTHON_INTERPRETER) $(TRAIN_MODULE) --config $$cfg --seed $$s; \
		done; \
	done

## Linear probe all BACKBONES with SEEDS list
## Example: make probe_all SEEDS="0 1 2"
.PHONY: probe_all
probe_all:
	@set -e; \
	for cfg in $(BACKBONES); do \
		for s in $(SEEDS); do \
			echo "=== Linear probe $$cfg seed=$$s ==="; \
			$(PYTHON_INTERPRETER) $(TRAIN_MODULE) --config $$cfg --linear_probe 1 --seed $$s; \
		done; \
	done

## Finetune all BACKBONES with SEEDS list
## Example: make finetune_all SEEDS="0 1 2"
.PHONY: finetune_all
finetune_all:
	@set -e; \
	for cfg in $(BACKBONES); do \
		for s in $(SEEDS); do \
			echo "=== Finetune $$cfg seed=$$s ==="; \
			$(PYTHON_INTERPRETER) $(TRAIN_MODULE) --config $$cfg --linear_probe 0 --seed $$s; \
		done; \
	done

#################################################################################
# Testing AUTOMATION                                                           #
#################################################################################
.PHONY: eval-best
eval-best:
	python -m src.eval_best_models 


#################################################################################
# SELF DOCUMENTING COMMANDS                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
