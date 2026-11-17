"""Malformed training data must fail before fitting."""
import pandas as pd
import pytest

from training.config import DATA_PATH
from training.data import load_data


@pytest.mark.parametrize("mutation", ["missing", "extra", "category", "negative", "infinite", "duplicate",
                                     "target", "invalid_charge", "missing_charge", "blank_nonzero", "fractional"])
def test_reject_invalid_dataset(tmp_path, mutation):
    frame = pd.read_csv(DATA_PATH).head(100).copy()
    if mutation == "missing":
        frame = frame.drop(columns="Contract")
    elif mutation == "extra":
        frame["future_outcome"] = 1
    elif mutation == "category":
        frame.loc[0, "Contract"] = "Weekly"
    elif mutation == "negative":
        frame.loc[0, "MonthlyCharges"] = -1
    elif mutation == "infinite":
        frame.loc[0, "MonthlyCharges"] = float("inf")
    elif mutation == "duplicate":
        frame.loc[1, "customerID"] = frame.loc[0, "customerID"]
    elif mutation == "target":
        frame.loc[0, "Churn"] = "Maybe"
    elif mutation == "invalid_charge":
        frame.loc[0, "TotalCharges"] = "not-a-number"
    elif mutation == "missing_charge":
        frame.loc[0, "TotalCharges"] = None
    elif mutation == "blank_nonzero":
        frame.loc[0, "tenure"] = 12
        frame.loc[0, "TotalCharges"] = " "
    else:
        frame["tenure"] = frame["tenure"].astype(float)
        frame.loc[0, "tenure"] = 1.5
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_data(path)


def test_documented_zero_tenure_blank_is_accepted(tmp_path):
    frame = pd.read_csv(DATA_PATH).head(100).copy()
    frame.loc[0, "tenure"] = 0
    frame.loc[0, "TotalCharges"] = " "
    path = tmp_path / "valid.csv"
    frame.to_csv(path, index=False)
    features, _ = load_data(path)
    assert features.loc[0, "TotalCharges"] == 0
