# src/config.py

import os
from dataclasses import dataclass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)


@dataclass
class DatasetConfig:
    name: str
    train_path: str
    test_path: str
    target_col: str
    id_col: str | None
    test_size: float = 0.2
    random_state: int = 42
    positive_label: int = 1  # assume binary 0/1


CREDIT_CONFIG = DatasetConfig(
    name="credit_scoring",
    train_path=os.path.join(DATA_DIR, "credit", "credit_train.csv"),
    test_path=os.path.join(DATA_DIR, "credit", "credit_test.csv"),
    target_col="default",      # <-- CHANGE to your target column
    id_col="client_id",        # <-- or None if no ID column
    test_size=0.2,
    random_state=42,
    positive_label=1,
)

FRAUD_CONFIG = DatasetConfig(
    name="fraud_detection",
    train_path=os.path.join(DATA_DIR, "fraud", "fraud_train.csv"),
    test_path=os.path.join(DATA_DIR, "fraud", "fraud_test.csv"),
    target_col="is_fraud",     # <-- CHANGE to your target column
    id_col="transaction_id",   # <-- or None if no ID column
    test_size=0.2,
    random_state=42,
    positive_label=1,
)
