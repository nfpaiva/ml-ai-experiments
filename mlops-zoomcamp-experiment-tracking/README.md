# MLOps Zoomcamp Experiment Tracking

This repository contains the `mlops-zoomcamp-experiment-tracking` application, a project designed for experiment tracking, model management, and hyperparameter optimization as part of the MLOps Zoomcamp course. The application leverages MLflow, Optuna, and scikit-learn to provide a robust workflow for machine learning experimentation and model lifecycle management.

## Features
- Experiment tracking with MLflow
- Hyperparameter optimization using Optuna
- Model training and evaluation with scikit-learn
- Data handling and preprocessing utilities
- Reproducible environment setup

## Installation

Follow these steps to set up the project in a new conda environment:

### 1. Create a new conda environment
```bash
conda create -n mlops_env python=3.8 -y
```

### 2. Activate the conda environment
```bash
conda activate mlops_env
```

### 3. Install the package and its dependencies
Make sure you are in the `mlops-zoomcamp-experiment-tracking` directory (where `pyproject.toml` is located):
```bash
pip install .
```

This will install all required dependencies as specified in the `pyproject.toml` file.

## Development dependencies
To install development dependencies (for testing and code formatting):
```bash
pip install .[dev]
```

## Usage
Refer to the source code and scripts in the repository for usage examples and entry points for training, experiment tracking, and model management.

---

For more details, see the code and documentation in this repository.
