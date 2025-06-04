# src/infer.py

import os
from typing import Literal

import joblib
import pandas as pd

from .config import MODELS_DIR


TaskType = Literal["credit", "fraud"]


def load_model(task: TaskType):
    model_filename = {
        "credit": "credit_model.joblib",
        "fraud": "fraud_model.joblib",
    }[task]
    path = os.path.join(MODELS_DIR, model_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Train it first.")
    return joblib.load(path)

def score_file(
    task: TaskType,
    input_csv_path: str,
    output_csv_path: str,
    id_col: str | None = None,
):
    model = load_model(task)
    df = pd.read_csv(input_csv_path).copy()

    ids = df[id_col] if id_col and id_col in df.columns else None
    if id_col and id_col in df.columns:
        X = df.drop(columns=[id_col])
    else:
        X = df

    y_prob = (
        model.predict_proba(X)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )
    y_pred = model.predict(X)

    result = pd.DataFrame()
    if ids is not None:
        result[id_col] = ids
    result["prediction"] = y_pred
    if y_prob is not None:
        result["probability"] = y_prob

    result.to_csv(output_csv_path, index=False)
    print(f"Saved predictions to {output_csv_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inference for credit/fraud models")
    parser.add_argument(
        "--task",
        type=str,
        choices=["credit", "fraud"],
        required=True,
        help="Which model to use",
    )
    parser.add_argument("--input", type=str, required=True, help="Input CSV path")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--id-col", type=str, default=None, help="Optional ID column")

    args = parser.parse_args()
    score_file(
        task=args.task,
        input_csv_path=args.input,
        output_csv_path=args.output,
        id_col=args.id_col,
    )