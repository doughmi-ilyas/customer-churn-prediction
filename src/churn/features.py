"""Feature engineering shared by the Streamlit app, the API and the tests."""
import pandas as pd

SERVICES = ["PhoneService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


def prepare(data: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered columns the model was trained on."""
    out = data.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(0.0)
    out["charges_per_tenure"] = out["TotalCharges"] / (out["tenure"] + 1)
    out["is_new_customer"] = (out["tenure"] <= 12).astype(int)
    out["is_month_to_month"] = (out["Contract"] == "Month-to-month").astype(int)
    out["is_fiber"] = (out["InternetService"] == "Fiber optic").astype(int)
    out["is_electronic_check"] = (out["PaymentMethod"] == "Electronic check").astype(int)
    out["nb_services"] = (out[SERVICES] == "Yes").sum(axis=1)
    out["tenure_band"] = pd.cut(out["tenure"], bins=[-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"]).astype(str)
    return out
