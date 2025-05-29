# src/preprocessing.py

from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def detect_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Detect numeric and categorical columns from a training dataframe."""
    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return numeric_features, categorical_features

def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor

def build_credit_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Pipeline for credit scoring – Logistic Regression."""
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    clf = LogisticRegression(
        max_iter=200,
        class_weight="balanced",  # often helpful in credit default data
        n_jobs=None,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", clf),
        ]
    )
    return pipeline


def build_fraud_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Pipeline for fraud detection – RandomForest."""
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",  # fraud datasets are highly imbalanced
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", clf),
        ]
    )
    return pipeline