import argparse
import logging
import os
from pathlib import Path
import yaml

import mlflow
import pandas as pd
from mlflow.pyfunc import load_model

from datahandler import DataHandler
from mlflowhelper import MlFlowModelManager
from mlflowhelper import MlFlowContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Inference Pipeline")
    parser.add_argument(
        "--month",
        type=str,
        required=True,
        help="Month for inference in the format YYYY-MM",
    )
    args = parser.parse_args()

    # Constants
    FILE_PATH = Path(__file__).resolve()
    BASE_PATH = FILE_PATH.parent
    config_path = os.path.join(BASE_PATH, "config.yaml")

    with open(config_path, "r", encoding="utf-8") as filename:
        config = yaml.safe_load(filename)

    DATA_DIR = Path("data/")
    INFERENCE_DIR = DATA_DIR / "inference"
    S3_URL = config["constants"]["S3_URL"]
    HPO_EXPERIMENT_NAME = config["constants"]["HPO_EXPERIMENT_NAME"]

    # Ensure inference directory exists
    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize DataHandler
    data_handler = DataHandler(DATA_DIR, S3_URL)

    # Download and preprocess the dataset
    parquet_file = f"green_tripdata_{args.month}.parquet"
    file_path = data_handler.download_data(parquet_file)
    dataset = pd.read_parquet(file_path)
    preprocessed_data = data_handler.preprocess_dataset(dataset)

    # Separate features (X) and labels (y)
    X = preprocessed_data.drop(columns=["is_cash_payment"])
    y = preprocessed_data["is_cash_payment"]

    # Create a mock args object with the required attributes
    class ArgsMock:
        flag_reset_mlflow = "N"

    args_mock = ArgsMock()

    # Connect to MLflow and retrieve the champion model
    # Initialize MLflow context
    db_path = Path(__file__).resolve().parent.parent / "mlflow" / "mlflow.db"
    mlflow_context = MlFlowContext(db_path, HPO_EXPERIMENT_NAME, args_mock)

    # Initialize MLflow model manager with context
    model_manager = MlFlowModelManager(mlflow_context)

    # Retrieve the URI of the model registered as "Production" in MLflow
    champion_model_uri = model_manager.get_production_model_uri("mlops-zoomcamp-champion-model")

    # Load the champion model
    logger.info("Loading champion model from MLflow...")
    model = load_model(champion_model_uri)


    # Access the raw sklearn model
    logger.info("Accessing the raw sklearn model...")
    raw_model = model.get_raw_model()
    logger.info(f"Raw model type: {type(raw_model)}")
    logger.info(f"Raw model attributes: {dir(raw_model)}")

    # Perform inference
    logger.info("Performing inference...")
    predictions = raw_model.predict(X)

    # Attempt to access predict_proba or similar method from the raw model
    if hasattr(raw_model, 'predict_proba'):
        probabilities = raw_model.predict_proba(X)[:, 1]  # Assuming binary classification
    else:
        logger.warning("The raw model does not have a 'predict_proba' method. Probabilities will not be included.")
        probabilities = None

    # Save the results along with features
    output_file = INFERENCE_DIR / f"inference_{args.month}.csv"
    results = pd.concat([X.reset_index(drop=True), pd.DataFrame({"y_true": y, "y_pred": predictions, "y_proba": probabilities})], axis=1)
    results.to_csv(output_file, index=False)
    logger.info("Inference results saved to %s", output_file)

if __name__ == "__main__":
    main()
