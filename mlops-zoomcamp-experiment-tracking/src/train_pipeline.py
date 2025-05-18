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


def run_optimization(
    file_names: Dict[str, Dict[str, str]],
    dataset_split_opt: DatasetSplit,
    mlflow_exp_obj: MlFlowExperimentRegistry,
    num_trials: int,
    pipeline_opt: Pipeline,
) -> None:
    """Run the hyperparameter optimization using Optuna.

    Args:
        dataset_split_opt (DatasetSplit): A data class containing the split datasets and labels.
        client (MlflowClient): MLflow tracking self.client_mlflow.
        experiment (mlflow.entities.Experiment): MLflow experiment.
        num_trials (int): Number of hyperparameter optimization trials.
        pipeline_opt (Pipeline): pipeline for the model.

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
        # Define hyperparameters for ExplainableBoostingClassifier
        params = {
            "max_bins": trial.suggest_int("max_bins", 32, 64),
            "max_interaction_bins": trial.suggest_int("max_interaction_bins", 32, 64),
            "interactions": trial.suggest_int("interactions", 0, 1),
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
    DATA_DIR = (BASE_PATH / config["constants"]["DATA_DIR"]).resolve()
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
        ("ebm", ExplainableBoostingClassifier())
    ])

    # Instantiate mlflow context object
    db_path = os.path.join(BASE_PATH, "mlflow.db")
    mlflowcontext = MlFlowContext(db_path, HPO_EXPERIMENT_NAME, args)

    # Add debug logs to track pipeline execution
    logger.info("Starting the pipeline execution...")

    # Log dataset split details
    logger.info(f"Dataset split details: {dataset_split}")

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
