"""Loads the versioned artifact once and turns a validated customer into a score + risk tier."""
import json

import joblib
import pandas as pd

from backend.app.schemas import Customer
from training.evaluation import risk_tier


class Predictor:
    def __init__(self, models_dir):
        artifacts = sorted(models_dir.glob("*.joblib"))
        if not artifacts:
            raise FileNotFoundError(f"No model artifact in {models_dir}")
        bundle = joblib.load(artifacts[-1])
        self.pipeline = bundle["pipeline"]
        self.metadata = bundle["metadata"]

        schema = self.metadata["feature_schema"]
        expected = set(schema["numerical"]) | set(schema["categorical"])
        if expected != set(Customer.model_fields):
            raise ValueError("API schema does not match the model's feature schema")

        log_path = models_dir / "experiment_log.json"
        self.experiments = json.loads(log_path.read_text()) if log_path.exists() else []

    @property
    def version(self):
        return self.metadata["model_version"]

    def predict(self, customer: Customer):
        row = pd.DataFrame([customer.model_dump()])
        probability = float(self.pipeline.predict_proba(row)[0, 1])
        return probability, risk_tier(probability, self.metadata["risk_tiers"])
