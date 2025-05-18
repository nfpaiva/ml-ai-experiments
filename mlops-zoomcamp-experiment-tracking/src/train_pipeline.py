"""
train_pipeline.py

This script orchestrates the training pipeline for machine learning models. It includes data loading,
preprocessing, model training, hyperparameter optimization, and model evaluation. The results are
logged in MLflow for experiment tracking and reproducibility.

Usage:
    python train_pipeline.py --train-month YYYY-MM --val-month YYYY-MM --test-month YYYY-MM

Arguments:
    --train-month (str): The month for the training dataset in the format YYYY-MM.
    --val-month (str): The month for the validation dataset in the format YYYY-MM.
    --test-month (str): The month for the test dataset in the format YYYY-MM.
    --num-trials (int, optional): Number of trials for hyperparameter optimization. Default is 1.
    --flag-reset-mlflow (str, optional): Whether to reset the MLflow database. Options are 'Y' or 'N'. Default is 'N'.

Functions:
    run_optimization(): Runs hyperparameter optimization using Optuna.
"""

"""train_pipeline.py """
import argparse
import datetime
import logging
import os
from pathlib import Path
from typing import Dict


import yaml

import optuna
import mlflow  # type: ignore
from optuna.samplers import TPESampler
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline  # type: ignore

from datasetsplit import DatasetSplit
from datahandler import DataHandler
from mlflowhelper import MlFlowContext, MlFlowExperimentRegistry, MlFlowModelManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get absolute path of this module and its directory path
FILE_PATH = Path(__file__).resolve()
BASE_PATH = FILE_PATH.parent

# Add the scripts directory to the Python module search path
import sys

sys.path.append(str(BASE_PATH.parent / "scripts"))

# Rebuild the TRACKING_URI and ARTIFACT_ROOT using relative paths
TRACKING_URI = f"sqlite:///{BASE_PATH.parent / 'mlflow/mlflow.db'}"
ARTIFACT_ROOT = str(BASE_PATH.parent / 'mlflow/mlruns')

# Set MLflow tracking URI
mlflow.set_tracking_uri(TRACKING_URI)


def run_optimization(
    file_names: Dict[str, Dict[str, str]],
    dataset_split_opt: DatasetSplit,
    mlflow_exp_obj: MlFlowExperimentRegistry,
    num_trials: int,
    pipeline_opt: Pipeline,
) -> None:
    """Run the hyperparameter optimization using Optuna.

    Args:
        file_names (Dict[str, Dict[str, str]]): Dictionary containing file paths for datasets.
        dataset_split_opt (DatasetSplit): A data class containing the split datasets and labels.
        mlflow_exp_obj (MlFlowExperimentRegistry): MLflow experiment registry object.
        num_trials (int): Number of hyperparameter optimization trials.
        pipeline_opt (Pipeline): Scikit-learn pipeline for the model.

    Returns:
        None
    """

    logger.info("Loading and splitting the dataset...")

    x_train_opt, x_val_opt, x_test_opt = (
        dataset_split_opt.x_train,
        dataset_split_opt.x_val,
        dataset_split_opt.x_test,
    )
    y_train_opt, y_val_opt, y_test_opt = (
        dataset_split_opt.y_train,
        dataset_split_opt.y_val,
        dataset_split_opt.y_test,
    )

    # Build HPO Run id
    hpo_run_id = mlflow_exp_obj.get_max_hpo_run_id()

    def objective(trial) -> float:
        """
        Objective function for Optuna hyperparameter optimization.

        Args:
            trial (optuna.trial.Trial): A single trial object for suggesting hyperparameters.

        Returns:
            float: The F1 score on the validation dataset, used as the optimization metric.

        This function defines the hyperparameters for the ExplainableBoostingClassifier,
        trains the model, evaluates it on the validation dataset, and logs the results
        to MLflow.
        """

        # Define hyperparameters for ExplainableBoostingClassifier
        params = {
            "max_bins": trial.suggest_int("max_bins", 8, 16),
            "max_interaction_bins": trial.suggest_int("max_interaction_bins", 8, 16),
            "interactions": trial.suggest_int("interactions", 0, 0),  # Disable interactions
        }

        with mlflow.start_run(experiment_id=mlflow_exp_obj.experiment.experiment_id):
            # Log run metadata
            params_str = "_".join(
                [f"{param}={value}" for param, value in params.items()]
            )
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            run_name = f"{params_str}_{timestamp}"
            mlflow.set_tag("mlflow.runName", run_name)
            mlflow.set_tag("developer", "nfpaiva")
            mlflow.set_tag("hpo_run_id", str(hpo_run_id))
            mlflow.log_params(params)

            # Concatenate train, validation, and test months into a single dataset tag
            dataset_tag = f"train:{args.train_month}, val:{args.val_month}, test:{args.test_month}"
            mlflow.set_tag("dataset", dataset_tag)

            # Tag the "Dataset" column with the same value as the "dataset" column
            mlflow.log_param("Dataset", dataset_tag)

            # Update pipeline with hyperparameters
            pipeline_opt.set_params(ebm__max_bins=params["max_bins"], ebm__max_interaction_bins=params["max_interaction_bins"], ebm__interactions=params["interactions"])

            # Train the model
            pipeline_opt.fit(x_train_opt, y_train_opt)

            # Evaluate on validation and test sets
            y_pred_val = pipeline_opt.predict(x_val_opt)
            y_pred_test = pipeline_opt.predict(x_test_opt)

            # Calculate classification metrics
            accuracy_val = accuracy_score(y_val_opt, y_pred_val)
            precision_val = precision_score(y_val_opt, y_pred_val)
            recall_val = recall_score(y_val_opt, y_pred_val)
            f1_val = f1_score(y_val_opt, y_pred_val)

            # Log metrics to MLflow
            mlflow.log_metric("accuracy_val", accuracy_val)
            mlflow.log_metric("precision_val", precision_val)
            mlflow.log_metric("recall_val", recall_val)
            mlflow.log_metric("f1_val", f1_val)

            # Log the model
            mlflow.sklearn.log_model(sk_model=pipeline_opt, artifact_path="model")

        return f1_val

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=num_trials)


if __name__ == "__main__":
    # Get absolute path of this module and its directory path
    FILE_PATH = Path(__file__).resolve()
    BASE_PATH = FILE_PATH.parent

    config_path = os.path.join(BASE_PATH, "config.yaml")

    with open(config_path, "r", encoding="utf-8") as filename:
        config = yaml.safe_load(filename)

    # attributing required constants
    DATA_DIR = (BASE_PATH.parent / "data").resolve()
    S3_URL = config["constants"]["S3_URL"]
    PREFIX = config["constants"]["PREFIX"]
    HPO_EXPERIMENT_NAME = config["constants"]["HPO_EXPERIMENT_NAME"]
    HPO_BEST_MODEL = config["constants"]["HPO_BEST_MODEL"]
    HPO_CHAMPION_MODEL = config["constants"]["HPO_CHAMPION_MODEL"]

    parser = argparse.ArgumentParser(description="MLOPS Zoomcamp Homework 2")
    parser.add_argument(
        "--train-month",
        type=str,
        required=True,
        help="Train month in format YYYY-MM",
    )
    parser.add_argument(
        "--val-month",
        type=str,
        required=True,
        help="Validation month in format YYYY-MM",
    )
    parser.add_argument(
        "--test-month",
        type=str,
        required=True,
        help="Test month in format YYYY-MM",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,  # Changed from 2 to 1
        help="Number of trials for each Run at mlflow",
    )
    parser.add_argument(
        "--flag-reset-mlflow",
        type=str,
        default="N",
        help="Flag to ask mlflow reset - deleting DB and mlruns artifacts folders. "
        "Y=reset, N=no reset",
    )
    args = parser.parse_args()

    FILE_NAMES = {
        "train": {"month": args.train_month},
        "val": {"month": args.val_month},
        "test": {"month": args.test_month},
    }

    # Instantiate the DataHandler object
    data_handler = DataHandler(DATA_DIR, S3_URL)

    # Download the train, validation, and test data and update the dictionary
    # with corresponding file_paths required for splitting datasets function
    for file_type, month_info in FILE_NAMES.items():
        month = month_info["month"]
        file_name = PREFIX + month + ".parquet"
        month_info["file_name"] = file_name
        month_info["link"] = data_handler.download_data(file_name)

    # Load and split the dataset using the same data_handler instance
    dataset_split = data_handler.load_split_dataset(FILE_NAMES)

    # Pipeline to use ExplainableBoostingClassifier
    pipeline = Pipeline([
        ("ebm", ExplainableBoostingClassifier(max_bins=16, max_interaction_bins=16, interactions=0))
    ])

    # Ensure the reset flag is explicitly checked before passing to MlFlowContext
    reset_flag = args.flag_reset_mlflow.upper() == "Y"

    # Instantiate mlflow context object
    db_path = TRACKING_URI.replace("sqlite://", "")
    mlflowcontext = MlFlowContext(db_path, HPO_EXPERIMENT_NAME, args)

    # Add debug logs to track pipeline execution
    logger.info("Starting the pipeline execution...")

    # Log pipeline initialization
    logger.info("Pipeline initialized with ExplainableBoostingClassifier.")

    # Log MLflow context initialization
    logger.info(f"MLflow context initialized with database path: {db_path} and experiment name: {HPO_EXPERIMENT_NAME}")

    logger.info("Running the optimization...")

    run_optimization(
        FILE_NAMES,
        dataset_split,
        MlFlowExperimentRegistry(mlflowcontext),
        args.num_trials,
        pipeline,
    )

    # Log before running optimization
    logger.info("Starting hyperparameter optimization...")

    # Log after optimization
    logger.info("Hyperparameter optimization completed.")

    logger.info("Running the model registry...")

    # Log before model registration
    logger.info("Starting model registration...")

    MlFlowModelManager(mlflowcontext).run_register_model(
        HPO_CHAMPION_MODEL, dataset_split.x_test, dataset_split.y_test
    )

    # Log after model registration
    logger.info("Model registration completed.")
