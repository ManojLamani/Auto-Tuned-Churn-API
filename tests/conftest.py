import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.monitoring import RateLimiter


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        app.state.limiter = RateLimiter(100_000)  # tests send many requests; 429 is tested separately
        yield c


@pytest.fixture
def customer():
    return {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5, "TotalCharges": 190.0,
    }
