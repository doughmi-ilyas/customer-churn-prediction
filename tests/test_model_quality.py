"""Regression guard: the saved model must still reach the documented quality on the raw data."""
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from churn.service import ChurnService
from churn.features import prepare

ROOT = Path(__file__).resolve().parents[1]


def test_saved_model_roc_auc_on_dataset():
    df = pd.read_csv(ROOT / "data" / "raw" / "data.csv")
    y = (df.pop("Churn") == "Yes").astype(int)
    svc = ChurnService()
    X = prepare(df.drop(columns=["customerID"]))
    auc = roc_auc_score(y, svc.churn_model.predict_proba(X)[:, 1])
    # includes training rows, so this is only a sanity floor, not a test-set metric
    assert auc > 0.80
