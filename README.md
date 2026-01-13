# Chest X-ray Classification

## Overview
Chest X-ray classification project for Normal / Pneumonia / Tuberculosis
using classical features (HOG) and deep learning (CNNs, Transformers).

## Project Structure
├── LICENSE
├── Makefile
├── README.md
├── environment.yml        <- Conda environment for reproducibility
├── pyproject.toml
│
├── configs/               <- YAML configs for experiments and models
│   ├── base.yaml
│   ├── cnn.yaml
│   ├── resnet50.yaml
│   ├── densenet121.yaml
│   ├── efficientnet_b0.yaml
│   └── swin_tiny_patch4_window7_224.yaml
│
├── data/
│   ├── raw/               <- Original X-ray images (not tracked in Git)
│   ├── interim/           <- Intermediate preprocessing outputs
│   ├── processed/         <- Final model-ready images
│   ├── external/          <- Optional external datasets
│   ├── eda/               <- Dataset statistics and plots
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── src/
│   ├── data/              <- Dataset and DataLoader logic
│   ├── models/            <- Model builders (CNNs, ViTs)
│   ├── engine/            <- Training, evaluation, callbacks
│   ├── preprocessing/     <- Image preprocessing (e.g., CLAHE)
│   ├── utils/             <- Logging, metrics, reproducibility
│   ├── train.py
│   ├── train_CNN.py
│   ├── train_LinearProbe.py
│   ├── train_hog.py
│   └── eval_best_models.py
│
├── notebooks/             <- Exploratory data analysis and preprocessing
│   ├── 01.EDA.ipynb
│   │   ├── Dataset statistics and class distribution
│   │   ├── Image resolution and aspect ratio analysis
│   │   └── Identification of preprocessing issues
│   │
│   ├── 02.Preprocessing_Blackborder.ipynb
│   │   ├── Detection and removal of black borders
│   │   ├── Visualization of preprocessing effects
│   │   └── Impact on image geometry
│   │
│   ├── 03.EDA_FinalizePreprocessing.ipynb
│   │   ├── Post-preprocessing validation
│   │   ├── Final dataset statistics
│   │   └── Sanity checks before model training
│   │
│   ├── aspect_ratio_distribution.png
│   ├── image_resolution_distribution.png
│   ├── test_connection.py      <- MLflow / W&B connectivity test
│   │
│   ├── mlflow.db               <- Local MLflow tracking (not committed)
│   └── mlruns/                 <- MLflow runs (not committed)
├── reports/
│   ├── figures/           <- Confusion matrices, ROC, Grad-CAM
│   └── best_model_eval/   <- Final evaluation outputs
│
├── tests/                 <- Unit tests
├── docs/                  <- Project documentation (MkDocs)
└── references/            <- Background material and references

## Experiments
1. CNNs (ResNet, DenseNet, EfficientNet)
2. Linear probing
3. HOG + MLP / XGBoost / RandomForest baseline
4. Fine-tuning best model
5. Explainability (Grad-CAM / attention)

## Model Selection and Evaluation

We systematically select the best-performing models from multiple training paradigms and evaluate them on a shared held-out test set, while preserving each model’s native preprocessing pipeline. Final outputs are deployment-ready artifacts, including predictions, evaluation metrics, calibration analysis, and misclassification diagnostics.

The evaluated model families include:
- **CNN baselines** (end-to-end deep learning)
- **Linear Probe models** (frozen backbone with linear classifier)
- **HOG-based classical models** (MLP, Random Forest, XGBoost)

Model selection is driven entirely by **MLflow experiment tracking**, without manual configuration or post-hoc tuning, ensuring a fully reproducible and unbiased evaluation process.


## Reproducibility
```bash
conda env create -f environment.yml
conda activate chest-xray
python src/train.py --config configs/densenet121.yaml
```

## Disclaimer

This project is for research and educational purposes only.
It is not intended for clinical use, diagnosis, or medical decision-making.
