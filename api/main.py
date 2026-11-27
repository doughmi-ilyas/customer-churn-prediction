"""FastAPI service: POST /predict returns churn probability and customer segment."""
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from churn.service import ChurnService  # noqa: E402

YesNo = Literal["Yes", "No"]


class Customer(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=72)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0, le=500)
    TotalCharges: float = Field(ge=0)


class Prediction(BaseModel):
    churn_probability: float
    churn: bool
    threshold: float
    cluster: int
    segment: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = ChurnService()
    yield


app = FastAPI(title="Customer churn API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": app.state.service.info["model"]}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    return app.state.service.predict(customer.model_dump())
