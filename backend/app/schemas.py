"""Request/response contracts. Field names and allowed values match the trained feature schema
(checked against the model artifact at startup)."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["No", "Yes"]
InternetAddon = Literal["No", "No internet service", "Yes"]


class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["No", "No phone service", "Yes"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"
    ]
    MonthlyCharges: float = Field(ge=0, le=1000)
    TotalCharges: float = Field(ge=0, le=100000)


class Prediction(BaseModel):
    predictionId: str
    churnProbability: float
    riskTier: Literal["HIGH", "MEDIUM", "LOW"]
    threshold: float
    modelVersion: str
    preprocessingVersion: str
    timestamp: str
