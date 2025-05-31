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