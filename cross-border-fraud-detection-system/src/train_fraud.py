# src/train_fraud.py

import os
import joblib

from .config import FRAUD_CONFIG, MODELS_DIR
from .data_loader import load_dataset, split_features_target, train_valid_split
from .preprocessing import detect_feature_types, build_fraud_pipeline
from .evaluate import evaluate_and_print


def main():
    print("Loading fraud data...")
    train_df, test_df = load_dataset(FRAUD_CONFIG)

    X_train_full, y_train_full = split_features_target(train_df, FRAUD_CONFIG)
    X_test, y_test = split_features_target(test_df, FRAUD_CONFIG)

    X_train, X_valid, y_train, y_valid = train_valid_split(
        X_train_full, y_train_full, FRAUD_CONFIG
    )

    numeric_features, categorical_features = detect_feature_types(X_train)
    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)
    pipeline = build_fraud_pipeline(numeric_features, categorical_features)

    print("Fitting fraud detection model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating on validation set...")
    y_valid_pred = pipeline.predict(X_valid)
    y_valid_prob = (
        pipeline.predict_proba(X_valid)[:, 1]
        if hasattr(pipeline.named_steps["model"], "predict_proba")
        else None
    )
    evaluate_and_print("Fraud Detection (validation)", y_valid, y_valid_pred, y_valid_prob)

    print("Evaluating on test set...")
    y_test_pred = pipeline.predict(X_test)
    y_test_prob = (
        pipeline.predict_proba(X_test)[:, 1]
        if hasattr(pipeline.named_steps["model"], "predict_proba")
        else None
    )
    evaluate_and_print("Fraud Detection (test)", y_test, y_test_pred, y_test_prob)
    model_path = os.path.join(MODELS_DIR, "fraud_model.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Saved fraud model to {model_path}")


if __name__ == "__main__":
    main()