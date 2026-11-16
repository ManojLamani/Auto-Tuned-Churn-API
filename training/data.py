"""Load, validate and split the Telco churn dataset."""
import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split

from training.config import DATA_PATH, SEED, TEST_SIZE


def dataset_version(path=DATA_PATH):
    return "telco_churn-" + hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def load_data(path=DATA_PATH):
    """Return features X and target y (1 = churn), with basic validation."""
    df = pd.read_csv(path)

    # Blank TotalCharges only occur for tenure 0 customers, who have not been billed yet.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    assert df["customerID"].is_unique, "Duplicate customer IDs"
    assert set(df["Churn"].unique()) == {"Yes", "No"}, "Unexpected target values"
    assert (df[["tenure", "MonthlyCharges", "TotalCharges"]] >= 0).all().all(), "Negative values"

    X = df.drop(columns=["Churn", "customerID"])
    y = (df["Churn"] == "Yes").astype(int)
    return X, y


def split(X, y, seed=SEED):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)
