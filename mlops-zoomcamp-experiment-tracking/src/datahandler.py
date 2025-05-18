"""
datahandler.py

Module containing the DataHandler class for data management, preprocessing, and loading.

Example:
    # Create a DataHandler object
    handler = DataHandler(data_dir=Path("data/"), s3_url="https://s3.example.com/")

    # Download, preprocess, and load datasets
    parquet_file = "sample_dataset.parquet"
    file_path = handler.download_data(parquet_file)
    preprocessed_data = handler.preprocess_dataset(pd.read_parquet(file_path))
    dataset_split = handler.load_split_dataset(file_names_load)
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from datasetsplit import DatasetSplit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataHandler:
    """
    A class for handling data download, preprocessing, and loading operations.

    This class provides methods for downloading data from an S3 bucket, preprocessing datasets,
    and loading datasets. It also offers the functionality to load a split dataset.

    Attributes:
        data_dir (Path): The directory where data files are stored.
        s3_url (str): The base URL of the S3 bucket where data is hosted.

    Example:
        # Create a DataHandler object
        handler = DataHandler(data_dir=Path("data/"), s3_url="https://s3.example.com/")

        # Download a data file from S3
        file_path = handler.download_data("dataset.parquet")

        # Preprocess a dataset
        dataset = pd.read_parquet("dataset.parquet")
        preprocessed_data = handler.preprocess_dataset(dataset)

        # Load and preprocess a split dataset
        file_names_load = {
            "train": {"link": "train.parquet"},
            "val": {"link": "val.parquet"},
            "test": {"link": "test.parquet"},
        }
        split_data = handler.load_split_dataset(file_names_load)
    """

    def __init__(self, data_dir: Path, s3_url: str) -> None:
        """Initialize a DataHandler object.

        Args:
            data_dir (Path): The directory where data files will be stored.
            s3_url (str): The base URL of the S3 bucket where data is hosted.
        """

        self.data_dir = data_dir
        self.s3_url = s3_url

    def download_data(self, parquet_file: str) -> Path:
        """Download a data file from the specified S3 bucket.

        Args:
            parquet_file (str): The name of the data file to download.

        Returns:
            Path: The path to the downloaded file.
        """

        file_path = self.data_dir / parquet_file
        url = self.s3_url + parquet_file
        logger.info("Attempting to download filename %s", url)
        if not os.path.isfile(file_path):
            logger.info("File does not exist, downloading from S3 bucket.")
            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            subprocess.run(["wget", "-O", file_path, url], check=True)
            logger.info("File downloaded successfully and saved at %s", file_path)
        else:
            logger.info("File already exists.")
        return file_path

    def preprocess_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the input dataset by adding new features and filtering rows.

        This method performs preprocessing operations on the input dataset to enhance its
        suitability for downstream tasks. It calculates the duration of each trip, filters out
        rows with unrealistic durations, and creates a concatenated feature representing the
        pickup and dropoff locations. Additionally, it converts specific columns to string
        data type for categorical analysis. It also adds features for pickup hour, pickup weekday,
        and trip duration in minutes.

        Args:
            dataset (pd.DataFrame): The input dataset containing raw trip data.

        Returns:
            pd.DataFrame: A preprocessed dataset with added features and filtered rows.
        """

        # Drop rows with essential missing values
        dataset = dataset.dropna(subset=[
            "payment_type", "fare_amount", "trip_distance",
            "lpep_pickup_datetime", "lpep_dropoff_datetime"
        ])

        # Target: is the trip marked as cash?
        dataset["is_cash_payment"] = (dataset["payment_type"] == 2).astype(int)

        # Feature engineering
        dataset["pickup_hour"] = dataset["lpep_pickup_datetime"].dt.hour
        dataset["pickup_weekday"] = dataset["lpep_pickup_datetime"].dt.weekday
        dataset["trip_duration_minutes"] = (
            dataset["lpep_dropoff_datetime"] - dataset["lpep_pickup_datetime"]
        ).dt.total_seconds() / 60

        # Remove invalid trips
        dataset = dataset[
            (dataset["fare_amount"] > 0) &
            (dataset["trip_distance"] > 0) &
            (dataset["trip_duration_minutes"] > 0)
        ]

        # Final feature set (post-trip only, no leakage)
        features = [
            "fare_amount", "trip_distance", "passenger_count",
            "pickup_hour", "pickup_weekday", "trip_duration_minutes",
            "RatecodeID", "VendorID", "trip_type",
            "PULocationID", "DOLocationID",
            "congestion_surcharge", "extra", "tolls_amount",
            "mta_tax", "improvement_surcharge"
        ]

        dataset = dataset[features + ["is_cash_payment"]]

        # Fill & cast
        dataset["RatecodeID"] = dataset["RatecodeID"].fillna(1).astype(int)
        dataset["VendorID"] = dataset["VendorID"].fillna(1).astype(int)
        dataset["trip_type"] = dataset["trip_type"].fillna(1).astype(int)
        dataset["PULocationID"] = dataset["PULocationID"].fillna(0).astype(int)
        dataset["DOLocationID"] = dataset["DOLocationID"].fillna(0).astype(int)
        dataset["congestion_surcharge"] = dataset["congestion_surcharge"].fillna(0)
        dataset["extra"] = dataset["extra"].fillna(0)
        dataset["tolls_amount"] = dataset["tolls_amount"].fillna(0)
        dataset["mta_tax"] = dataset["mta_tax"].fillna(0)
        dataset["improvement_surcharge"] = dataset["improvement_surcharge"].fillna(0)

        return dataset

    def load_dataset(
        self, file_names_load: Dict[str, Dict[str, str]]
    ) -> Dict[str, pd.DataFrame]:
        """
        Load datasets from file paths specified in the provided dictionary.

        Args:
            file_names_load (Dict[str, Dict[str, str]]): A dictionary containing dataset names
                as keys and corresponding file paths as values.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing loaded datasets.
        """

        df_data = {}
        for dataset, inner_month_info in file_names_load.items():
            file_path = inner_month_info["link"]
            df_data[dataset] = pd.read_parquet(file_path)

        return df_data

    def load_split_dataset(self, file_names: Dict[str, Dict[str, str]]) -> DatasetSplit:
        """
        Load and preprocess a split dataset.

        Args:
            file_names (Dict[str, Dict[str, str]]): A dictionary containing dataset names
                as keys and corresponding file paths as values.

        Returns:
            DatasetSplit: A DatasetSplit object containing split datasets and labels.
        """

        df_data = self.load_dataset(file_names)

        df_data_processed = {
            dataset: self.preprocess_dataset(df_data[dataset])
            for dataset in file_names.keys()
        }

        x_train_inner = df_data_processed["train"].drop(columns=["is_cash_payment"])
        x_val_inner = df_data_processed["val"].drop(columns=["is_cash_payment"])
        x_test_inner = df_data_processed["test"].drop(columns=["is_cash_payment"])


        y_train_inner = df_data_processed["train"]["is_cash_payment"]
        y_val_inner = df_data_processed["val"]["is_cash_payment"]
        y_test_inner = df_data_processed["test"]["is_cash_payment"]

        return DatasetSplit(
            x_train_inner,
            x_val_inner,
            x_test_inner,
            y_train_inner,
            y_val_inner,
            y_test_inner,
        )
