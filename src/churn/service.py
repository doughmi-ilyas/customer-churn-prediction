"""Model loading and prediction, independent of any web framework."""
import json
from pathlib import Path

import joblib
import pandas as pd

from churn.features import prepare

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"


class ChurnService:
    def __init__(self, artifacts: Path = ARTIFACTS):
        self.info = json.loads((artifacts / "model_info.json").read_text(encoding="utf-8"))
        self.churn_model = joblib.load(artifacts / "churn_model.joblib")
        self.cluster_model = joblib.load(artifacts / "cluster_model.joblib")

    def predict(self, record: dict) -> dict:
        X = prepare(pd.DataFrame([record]))
        proba = float(self.churn_model.predict_proba(X)[0, 1])
        cluster = int(self.cluster_model.predict(X)[0])
        return {
            "churn_probability": proba,
            "churn": proba >= self.info["threshold"],
            "threshold": self.info["threshold"],
            "cluster": cluster,
            "segment": self.info["clusters"][str(cluster)]["name"],
        }
