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
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.feature_extraction import DictVectorizer  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore
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
        pipeline_opt (Pipeline): Sklearn pipeline for the model.

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

    # Fit the preprocessor on the training data only
    # once to speed up training (instead of fitting the whole pipeline for each train iteration)
    # # Perhaps this can be achieved with caching option (?)
    pipeline_opt.named_steps["preprocessor"].fit(
        x_train_opt.to_dict(orient="records"), y_train_opt
    )

    # # Preprocess the training data
    x_train_preproc = pipeline_opt.named_steps["preprocessor"].transform(
        x_train_opt.to_dict(orient="records")
    )

    # Build HPO Run id
    hpo_run_id = mlflow_exp_obj.get_max_hpo_run_id()

    def objective(trial) -> float:
        params = {
            "random_forest__n_estimators": trial.suggest_int("n_estimators", 10, 50, 1),
            "random_forest__max_depth": trial.suggest_int("max_depth", 1, 20, 1),
            "random_forest__min_samples_split": trial.suggest_int(
                "min_samples_split", 2, 10, 1
            ),
            "random_forest__min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 1, 4, 1
            ),
            "random_forest__random_state": 42,
            "random_forest__n_jobs": -1,
        }

        # Update the hyperparameters of the model in the pipeline
        pipeline_opt.set_params(**params)

        with mlflow.start_run():
            # Customized run name based on model name and parameters
            params_str = "_".join(
                [f"{param}={value}" for param, value in params.items()]
            )
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            run_name = f"{params_str}_{timestamp}"
            mlflow.set_tag("mlflow.runName", run_name)

            mlflow.set_tag("developer", "nfpaiva")

            mlflow.set_tag("hpo_run_id", str(hpo_run_id))

            mlflow.log_params(params)

            # Train the model part of the pipeline with the preprocessed data
            pipeline_opt.named_steps["random_forest"].fit(x_train_preproc, y_train_opt)

            y_pred_val = pipeline_opt.predict(x_val_opt.to_dict(orient="records"))
            rmse_val: float = mean_squared_error(y_val_opt, y_pred_val, squared=False)

            y_pred_test = pipeline_opt.predict(x_test_opt.to_dict(orient="records"))
            rmse_test: float = mean_squared_error(
                y_test_opt, y_pred_test, squared=False
            )

            mlflow.log_metric("RMSE_Val", rmse_val)
            logger.info("Trial %s: RMSE_Val = %s", trial.number, rmse_val)

            mlflow.log_metric("RMSE_Test", rmse_test)
            logger.info("Trial %s: RMSE_Test = %s", trial.number, rmse_test)

            model_id = mlflow_exp_obj.get_max_model_id(hpo_run_id)

            mlflow.set_tag("model_id", str(model_id))

            mlflow.set_tag(
                "data_info", str({k: v["month"] for k, v in file_names.items()})
            )

            # log pipeline as an mlflow artifact
            mlflow.sklearn.log_model(sk_model=pipeline_opt, artifact_path="model")

        return rmse_test

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=num_trials)


if __name__ == "__main__":
    # Get absolute path of this module and its directory path
    FILE_PATH = Path(__file__).resolve()
    BASE_PATH = FILE_PATH.parent

    config_path = os.path.join(BASE_PATH, "config.yaml")

    with open(config_path, "r", encoding="utf-8") as filename:
        config = yaml.safe_load(filename)

    # attributing required constants
    DATA_DIR = BASE_PATH / "data"
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
        default=2,
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

    preprocessor = Pipeline(
        [("dict_vectorizer", DictVectorizer(sparse=False))], memory="preprocessor_cache"
    )

    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("random_forest", RandomForestRegressor())]
    )

    # Instantiate mlflow context object
    db_path = os.path.join(BASE_PATH, "mlflow.db")
    mlflowcontext = MlFlowContext(db_path, HPO_EXPERIMENT_NAME, args)

    logger.info("Running the optimization...")

    run_optimization(
        FILE_NAMES,
        dataset_split,
        MlFlowExperimentRegistry(mlflowcontext),
        args.num_trials,
        pipeline,
    )

    logger.info("Running the model registry...")

    MlFlowModelManager(mlflowcontext).run_register_model(
        HPO_CHAMPION_MODEL, dataset_split.x_test, dataset_split.y_test
    )
