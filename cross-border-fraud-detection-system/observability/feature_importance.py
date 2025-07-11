# src/feature_importance.py
import pandas as pd

def extract_feature_importance(model, feature_names):
    if hasattr(model, "coef_"):
        imp = model.coef_[0]
    elif hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        return None

    return (
        pd.DataFrame({"feature": feature_names, "importance": imp})
        .sort_values("importance", ascending=False)
    )
