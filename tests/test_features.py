import pandas as pd

from churn.features import prepare


def row(**kw):
    base = dict(tenure=5, TotalCharges="100", Contract="Month-to-month", InternetService="Fiber optic",
                PaymentMethod="Electronic check", PhoneService="Yes", OnlineSecurity="No", OnlineBackup="Yes",
                DeviceProtection="No", TechSupport="No", StreamingTV="Yes", StreamingMovies="No")
    base.update(kw)
    return pd.DataFrame([base])


def test_engineered_columns():
    out = prepare(row()).iloc[0]
    assert out["charges_per_tenure"] == 100 / 6
    assert out["is_new_customer"] == 1 and out["is_month_to_month"] == 1
    assert out["is_fiber"] == 1 and out["is_electronic_check"] == 1
    assert out["nb_services"] == 3
    assert out["tenure_band"] == "0-12"


def test_blank_total_charges_become_zero():
    assert prepare(row(TotalCharges=" ")).iloc[0]["TotalCharges"] == 0.0


def test_input_not_mutated():
    df = row()
    prepare(df)
    assert "nb_services" not in df.columns


def test_tenure_bands():
    bands = [prepare(row(tenure=t)).iloc[0]["tenure_band"] for t in (0, 12, 13, 24, 48, 49, 72)]
    assert bands == ["0-12", "0-12", "13-24", "13-24", "25-48", "49-72", "49-72"]
