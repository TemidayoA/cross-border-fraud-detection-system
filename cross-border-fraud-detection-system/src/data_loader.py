# src/data_loader.py

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DatasetConfig


def load_dataset(config: DatasetConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test CSVs."""
    train_df = pd.read_csv(config.train_path)
    test_df = pd.read_csv(config.test_path)
    return train_df, test_df