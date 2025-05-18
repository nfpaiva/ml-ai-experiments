"""
setup_mlflow.py

This script sets up the MLflow environment for experiment tracking. It ensures that the necessary
folders and database files are created, and starts the MLflow server for tracking experiments.

Usage:
    python setup_mlflow.py

Attributes:
    EXPERIMENT_NAME (str): The name of the MLflow experiment.
    MLFLOW_FOLDER (str): The path to the MLflow folder.
    MLFLOW_DB_PATH (str): The path to the MLflow SQLite database file.
    ARTIFACT_ROOT (str): The root directory for MLflow artifacts.
    TRACKING_URI (str): The URI for the MLflow tracking server.

Functions:
    main(): Sets up the MLflow environment and starts the MLflow server.
"""

import os
import mlflow
import subprocess
import logging
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Set these as needed for your project
EXPERIMENT_NAME = "mlops-zoomcamp-hpo-experiment"
# Path to mlflow.db in src/ directory
MLFLOW_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mlflow"))
MLFLOW_DB_PATH = os.path.join(MLFLOW_FOLDER, "mlflow.db")
ARTIFACT_ROOT = os.path.join(MLFLOW_FOLDER, "mlruns")
TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"

# Ensure the mlflow folder exists
os.makedirs(MLFLOW_FOLDER, exist_ok=True)

# Delete existing mlruns folder and mlflow.db if they exist
if os.path.exists(ARTIFACT_ROOT):
    logging.info(f"Deleting existing mlruns folder at {ARTIFACT_ROOT}")
    shutil.rmtree(ARTIFACT_ROOT)

if os.path.exists(MLFLOW_DB_PATH):
    logging.info(f"Deleting existing mlflow.db at {MLFLOW_DB_PATH}")
    os.remove(MLFLOW_DB_PATH)

def main():
    """
    Sets up the MLflow environment and starts the MLflow server.

    This function performs the following steps:
    1. Configures the MLflow tracking URI.
    2. Creates the MLflow experiment if it does not already exist.
    3. Deletes existing MLflow database and artifact directories if they exist.
    4. Starts the MLflow server as a subprocess.

    Returns:
        None
    """
    logging.info(f"Setting MLflow tracking URI to: {TRACKING_URI}")
    mlflow.set_tracking_uri(TRACKING_URI)
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
        logging.info(f"Experiment '{EXPERIMENT_NAME}' created.")
    else:
        logging.info(f"Experiment '{EXPERIMENT_NAME}' already exists.")

    # Start MLflow server
    logging.info(f"Starting MLflow server with SQLite DB at {MLFLOW_DB_PATH} and artifact root {ARTIFACT_ROOT}")
    logging.info("You can access the MLflow UI at http://127.0.0.1:5000")
    # Start server as a subprocess
    cmd = [
        "mlflow", "server",
        "--backend-store-uri", TRACKING_URI,
        "--default-artifact-root", ARTIFACT_ROOT,
        "--host", "127.0.0.1",
        "--port", "5000"
    ]
    logging.info(f"Running command: {' '.join(cmd)}")
    # Start the server in the foreground
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
