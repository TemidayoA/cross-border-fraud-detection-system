# src/validator.py
def validate_dataframe(df, name="dataset"):
    if df.empty:
        raise ValueError(f"{name} is empty")

    if df.isnull().mean().mean() > 0.5:
        raise ValueError(f"{name} contains more than 50% missing data")

    if df.duplicated().sum() > 0:
        print(f"Warning: {df.duplicated().sum()} duplicate rows detected")
