"""Loads the versioned artifact once and turns a validated customer into a score + risk tier."""
import os

import pandas as pd

from backend.app.schemas import Customer
from training.artifacts import load_release
from training.evaluation import risk_tier


class Predictor:
    def __init__(self, models_dir, version=None):
        bundle, self.experiments = load_release(models_dir, version or os.getenv("MODEL_VERSION"))
        self.pipeline = bundle["pipeline"]
        self.metadata = bundle["metadata"]

        schema = self.metadata["feature_schema"]
        expected = set(schema["numerical"]) | set(schema["categorical"])
        if expected != set(Customer.model_fields):
            raise ValueError("API schema does not match the model's feature schema")


    @property
    def threshold_version(self):
        return self.metadata.get("threshold_version", f"{self.version}:threshold")

    @property
    def version(self):
        return self.metadata["model_version"]

    def predict(self, customer: Customer):
        row = pd.DataFrame([customer.model_dump()])
        probability = float(self.pipeline.predict_proba(row)[0, 1])
        return probability, risk_tier(probability, self.metadata["risk_tiers"])
