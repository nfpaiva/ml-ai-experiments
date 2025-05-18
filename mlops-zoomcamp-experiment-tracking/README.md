# Monitoring System Mockup Challenge: MLOps Zoomcamp Experiment Tracking

This repository contains the `mlops-zoomcamp-experiment-tracking` application, a foundational project designed to support the development of a monitoring system mockup. The application provides tools for experiment tracking, model management, and hyperparameter optimization as part of the MLOps Zoomcamp course. It leverages MLflow, Optuna, and scikit-learn to create a robust workflow for machine learning experimentation and model lifecycle management.

## Features
- Experiment tracking with MLflow
- Hyperparameter optimization using Optuna
- Model training and evaluation with scikit-learn pipeline, leveraging the Explainable Boosting Machine (EBM) model for interpretable machine learning.
- Data handling and preprocessing utilities
- Reproducible environment setup

## Introduction

The goal of this application is to provide a baseline mock system to:
- Train machine learning models using Optuna for hyperparameter optimization.
- Register models in MLflow.
- Compare the best model (defined by F1-score) as the champion model.

The system handles the following scenarios:
1. If no model is registered, the best model from the training pipeline is promoted as the champion.
2. If a model is already registered, the challenger model is compared to the current champion. If the challenger is better, the current champion is archived, and the challenger is promoted.

The application creates a clean MLflow database and can be executed from the command line using a Python script. You can pass the datasets as arguments, for example:
```bash
python src/train_pipeline.py --train-month 2021-01 --val-month 2021-02 --test-month 2021-03
```

## Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/nfpaiva/ml-ai-experiments.git
```

2. Create a new Conda environment:
```bash
conda create -n mlops_env python=3.10 -y
```

3. Activate the Conda environment:
```bash
conda activate mlops_env
```

4. Navigate to the project directory:
```bash
cd mlops-zoomcamp-experiment-tracking/
```

5. Install the package and development dependencies:
```bash
pip install .[dev]
```

## Setup MLflow Database and Launch Service

1. Set up the MLflow database:
```bash
python scripts/setup_mlflow.py
```

2. Open a new terminal window and activate the environment:
```bash
conda activate mlops_env
```

3. Navigate to the project directory:
```bash
cd mlops-zoomcamp-experiment-tracking/
```

4. Train models for different months to test the model promotion mechanism:
```bash
python src/train_pipeline.py --train-month 2021-01 --val-month 2021-02 --test-month 2021-03
python src/train_pipeline.py --train-month 2021-02 --val-month 2021-03 --test-month 2021-04
python src/train_pipeline.py --train-month 2021-03 --val-month 2021-04 --test-month 2021-05
```

5. Check the MLflow UI at [http://localhost:5000/](http://localhost:5000/):
   - The first model will be added as the champion.
   - Only at the third model, the F1-score will be better, and the latter will be promoted to champion.

## Reference Notebook

For more context about the training task, refer to the notebook:
[green_taxi_cash_prediction_ebm.ipynb](notebooks/green_taxi_cash_prediction_ebm.ipynb)

## Task Briefing

The goal is to design and implement the following features on top of this system:
1. **Inference for a given month**: Implement a mechanism to perform inference on new data for a specific month.
2. **Capture inference results**: Store and manage the results of the inference process.
3. **Design monitoring use cases**: Identify and define monitoring scenarios to ensure model performance and data quality.
4. **Implement monitoring use cases**: Develop the monitoring use cases to track metrics and anomalies. This implementation can leverage different components, such as:
   - Metric calculation and visualization tools specific for monitoring models, like NannyML or Evidently.
   - Custom Python scripts for tailored monitoring needs.
   - Orchestration tools like Airflow for managing workflows.
   - Visualization tools like Grafana or Streamlit for presenting insights and dashboards.
5. **Create a demo**: Build a small demo to guide users through the workflow and UI, showcasing the system's capabilities.

### Example Use Case: Alert Implementation and Plan-B Mechanism

An example of a monitoring use case is implementing an alerting system to notify stakeholders when significant distribution changes are detected in important variables for the model. This can include:

- **Alerting System**: Automatically triggering alerts when data distribution drifts beyond a predefined threshold.
- **Fallback Mechanism (Plan-B)**: Handling such scenarios by:
  - Using inference results from the previous dataset if the new dataset is heavily affected by distribution changes.
  - Logging the issue and pausing further automated processes until the issue is resolved.

This example highlights the importance of designing alerts and fallback mechanisms based on monitoring use cases rather than solely focusing on technological exploration.

## Python Modules and Workflow

### Python Modules
The application is structured into several Python modules, each responsible for a specific part of the workflow:

1. **`datahandler.py`**: Handles data loading and preprocessing tasks, ensuring the datasets are clean and ready for training.
2. **`datasetsplit.py`**: Splits the data into training, validation, and test sets based on the provided configuration.
3. **`mlflowhelper.py`**: Manages MLflow operations, including model registration, comparison, and promotion to production.
4. **`train_pipeline.py`**: The main entry point for training models. It orchestrates the entire pipeline, from data loading to model evaluation.

### Program Workflow

1. **Data Preparation**:
   - The `datahandler.py` module loads the datasets and performs initial preprocessing.
   - The `datasetsplit.py` module splits the data into training, validation, and test sets.

2. **Model Training**:
   - The `train_pipeline.py` module trains the model using a scikit-learn pipeline.
   - The model is wrapped in a scikit-learn pipeline for consistent preprocessing and evaluation.

3. **Model Evaluation**:
   - The trained model is evaluated on the validation and test datasets.
   - Metrics such as F1-score are calculated to assess model performance.

4. **Model Registration and Promotion**:
   - The `mlflowhelper.py` module handles MLflow operations.
   - If no model is registered, the trained model is promoted as the champion.
   - If a model is already registered, the new model is compared to the current champion. If the new model performs better, it is promoted, and the current champion is archived.

5. **MLflow Integration**:
   - All experiments, metrics, and models are logged in MLflow for tracking and reproducibility.
   - The MLflow UI can be used to visualize and manage the experiments.

## Arguments for `train_pipeline.py`

When running the `train_pipeline.py` script, the following arguments can be used:

1. **`--train-month`** (required):
   - Specifies the month for the training dataset in the format `YYYY-MM`.
   - Example: `--train-month 2021-01`

2. **`--val-month`** (required):
   - Specifies the month for the validation dataset in the format `YYYY-MM`.
   - Example: `--val-month 2021-02`

3. **`--test-month`** (required):
   - Specifies the month for the test dataset in the format `YYYY-MM`.
   - Example: `--test-month 2021-03`

4. **`--num-trials`** (optional):
   - Specifies the number of trials for hyperparameter optimization.
   - Default: `1`
   - Example: `--num-trials 10`

5. **`--flag-reset-mlflow`** (optional):
   - Specifies whether to reset the MLflow database and artifact directories.
   - Options: `Y` (reset) or `N` (no reset).
   - Default: `N`
   - Example: `--flag-reset-mlflow Y`
