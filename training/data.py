"""Load, validate and split the Telco churn dataset."""
import hashlib

import pandas as pd
from pydantic import TypeAdapter, ValidationError
from sklearn.model_selection import train_test_split

from training.config import DATA_PATH, SEED, TEST_SIZE
from training.schema import Customer


def dataset_version(path=DATA_PATH):
    return "telco_churn-" + hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def load_data(path=DATA_PATH):
    """Validate the full serving schema before splitting or fitting anything."""
    df = pd.read_csv(path)
    expected = set(Customer.model_fields) | {"customerID", "Churn"}
    if set(df.columns) != expected:
        raise ValueError(f"Dataset columns mismatch; missing={sorted(expected - set(df.columns))}, "
                         f"unexpected={sorted(set(df.columns) - expected)}")
    if df.empty or df["customerID"].isna().any() or not df["customerID"].is_unique:
        raise ValueError("Empty dataset, missing or duplicate customer IDs")
    if set(df["Churn"].unique()) != {"Yes", "No"}:
        raise ValueError("Unexpected target values; both Yes and No are required")
    # Only the documented blank/zero-tenure case means the customer has not been billed.
    blank = df["TotalCharges"].astype(str).str.strip().eq("")
    if (blank & df["tenure"].ne(0)).any():
        raise ValueError("Blank TotalCharges is only valid for zero-tenure customers")
    try:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].mask(blank, "0"), errors="raise")
        for name in ("tenure", "SeniorCitizen", "MonthlyCharges", "TotalCharges"):
            if not pd.api.types.is_numeric_dtype(df[name]):
                raise ValueError(f"Invalid numeric column: {name}")
        TypeAdapter(list[Customer]).validate_python(df.drop(columns=["Churn", "customerID"]).to_dict("records"))
    except (ValueError, ValidationError) as exc:
        raise ValueError("Dataset violates customer feature schema") from exc

    X = df.drop(columns=["Churn", "customerID"])
    y = (df["Churn"] == "Yes").astype(int)
    return X, y


def split(X, y, seed=SEED):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)
