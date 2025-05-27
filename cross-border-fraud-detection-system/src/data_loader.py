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

def split_features_target(
    df: pd.DataFrame, config: DatasetConfig
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features and target; drop ID column if present."""
    df = df.copy()
    y = df[config.target_col]
    X = df.drop(columns=[config.target_col])
    if config.id_col is not None and config.id_col in X.columns:
        X = X.drop(columns=[config.id_col])
    return X, y

def train_valid_split(
    X: pd.DataFrame,
    y: pd.Series,
    config: DatasetConfig,
):
    return train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )