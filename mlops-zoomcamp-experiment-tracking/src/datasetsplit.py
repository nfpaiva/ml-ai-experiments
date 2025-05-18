"""
datasetsplit.py
This module defines a data class for encapsulating the split datasets and labels used 
in a machine learning pipeline. The class provides an organized way to store and access training, 
validation, and testing datasets along with their corresponding labels.
"""

from dataclasses import dataclass
from typing import Union
import pandas as pd
import numpy as np


@dataclass
class DatasetSplit:
    """A data class representing the split datasets and labels.

    This data class encapsulates the training, validation, and testing datasets along with their
    corresponding labels. It is designed to hold the following data:

    Attributes:
        x_train (pd.DataFrame): The pd DataFrame containing the features of the training dataset.
        x_val (pd.DataFrame): The pd DataFrame containing the features of the validation dataset.
        x_test (pd.DataFrame): The pd DataFrame containing the features of the testing dataset.
        y_train (np.ndarray): The NumPy array containing the labels of the training dataset.
        y_val (np.ndarray): The NumPy array containing the labels of the validation dataset.
        y_test (np.ndarray): The NumPy array containing the labels of the testing dataset.

    Example:
        # Create a DatasetSplit object
        dataset_split_opt = DatasetSplit(
            x_train=train_features,
            x_val=val_features,
            x_test=test_features,
            y_train=train_labels,
            y_val=val_labels,
            y_test=test_labels,
        )

        # Access the individual datasets and labels
        x_train_data = dataset_split_opt.x_train
        y_test_labels = dataset_split_opt.y_test
    """

    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train: Union[np.ndarray, pd.Series]
    y_val: Union[np.ndarray, pd.Series]
    y_test: Union[np.ndarray, pd.Series]
