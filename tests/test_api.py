import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

ROOT = Path(__file__).resolve().parents[1]
RISKY = dict(gender="Female", SeniorCitizen=0, Partner="No", Dependents="No", tenure=2, PhoneService="Yes",
             MultipleLines="No", InternetService="Fiber optic", OnlineSecurity="No", OnlineBackup="No",
             DeviceProtection="No", TechSupport="No", StreamingTV="No", StreamingMovies="No",
             Contract="Month-to-month", PaperlessBilling="Yes", PaymentMethod="Electronic check",
             MonthlyCharges=85.0, TotalCharges=170.0)
LOYAL = dict(RISKY, tenure=68, Contract="Two year", InternetService="DSL", OnlineSecurity="Yes", TechSupport="Yes",
             OnlineBackup="Yes", PaymentMethod="Bank transfer (automatic)", PaperlessBilling="No",
             MonthlyCharges=60.0, TotalCharges=4080.0, Partner="Yes", Dependents="Yes")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_shape_and_threshold(client):
    r = client.post("/predict", json=RISKY)
    assert r.status_code == 200
    body = r.json()
    info = json.loads((ROOT / "artifacts" / "model_info.json").read_text(encoding="utf-8"))
    assert 0 <= body["churn_probability"] <= 1
    assert body["threshold"] == info["threshold"]
    assert body["churn"] == (body["churn_probability"] >= body["threshold"])


def test_risky_customer_scores_higher_than_loyal(client):
    risky = client.post("/predict", json=RISKY).json()
    loyal = client.post("/predict", json=LOYAL).json()
    assert risky["churn_probability"] > loyal["churn_probability"]


def test_invalid_payload_rejected(client):
    assert client.post("/predict", json=dict(RISKY, Contract="Weekly")).status_code == 422
    assert client.post("/predict", json=dict(RISKY, tenure=-1)).status_code == 422
    missing = {k: v for k, v in RISKY.items() if k != "tenure"}
    assert client.post("/predict", json=missing).status_code == 422
