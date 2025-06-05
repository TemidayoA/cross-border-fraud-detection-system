# src/train_credit.py

import os
import joblib

from .config import CREDIT_CONFIG, MODELS_DIR
from .data_loader import load_dataset, split_features_target, train_valid_split
from .preprocessing import detect_feature_types, build_credit_pipeline
from .evaluate import evaluate_and_print


def main():
    print("Loading credit data...")
    train_df, test_df = load_dataset(CREDIT_CONFIG)

    X_train_full, y_train_full = split_features_target(train_df, CREDIT_CONFIG)
    X_test, y_test = split_features_target(test_df, CREDIT_CONFIG)

    # Split off validation from train
    X_train, X_valid, y_train, y_valid = train_valid_split(
        X_train_full, y_train_full, CREDIT_CONFIG
    )