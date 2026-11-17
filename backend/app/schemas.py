"""Serving contracts share the validated training feature schema."""
from typing import Literal

from pydantic import BaseModel

from training.schema import Customer as Customer


class Prediction(BaseModel):
    predictionId: str
    churnProbability: float
    riskTier: Literal["HIGH", "MEDIUM", "LOW"]
    threshold: float
    thresholdVersion: str
    modelVersion: str
    preprocessingVersion: str
    timestamp: str
