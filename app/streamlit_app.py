import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SERVICES = ["PhoneService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

LABELS = {
    "Yes": "Oui",
    "No": "Non",
    "Female": "Femme",
    "Male": "Homme",
    "Fiber optic": "Fibre optique",
    "Month-to-month": "Mensuel",
    "One year": "1 an",
    "Two year": "2 ans",
    "Electronic check": "Chèque électronique",
    "Mailed check": "Chèque papier",
    "Bank transfer (automatic)": "Virement (automatique)",
    "Credit card (automatic)": "Carte bancaire (automatique)",
}
ADDONS = {
    "OnlineSecurity": "Sécurité en ligne",
    "OnlineBackup": "Sauvegarde en ligne",
    "DeviceProtection": "Protection de l'appareil",
    "TechSupport": "Support technique",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming films",
}


def prepare(data):
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


def select(label, options):
    return st.selectbox(label, options, format_func=lambda v: LABELS.get(v, v))


@st.cache_resource
def load():
    art = ROOT / "artifacts"
    info = json.loads((art / "model_info.json").read_text(encoding="utf-8"))
    return joblib.load(art / "churn_model.joblib"), joblib.load(art / "cluster_model.joblib"), info


st.set_page_config(page_title="Prédiction du churn", page_icon="📉", layout="wide")
churn_model, cluster_model, info = load()

st.title("Prédiction du churn client")
st.caption("Saisissez les caractéristiques d'un client pour obtenir son profil (cluster) et son risque de résiliation.")

with st.sidebar:
    st.header("Modèle")
    st.write(f"**{info['model']}** {info['params']}")
    st.write(f"Seuil de décision : **{info['threshold']}**")
    m = info["metrics"]
    st.write(f"Test : ROC-AUC {m['test_roc_auc']:.3f} · Recall {m['test_recall']:.2f} · Precision {m['test_precision']:.2f} · F1 {m['test_f1']:.2f}")

with st.form("client"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Client")
        gender = select("Genre", ["Female", "Male"])
        senior = st.checkbox("Senior (65 ans et plus)")
        partner = select("Partenaire", ["No", "Yes"])
        dependents = select("Personnes à charge", ["No", "Yes"])
        tenure = st.slider("Ancienneté (mois)", 0, 72, 12)
    with c2:
        st.subheader("Services")
        phone = select("Téléphone", ["Yes", "No"])
        multiple = select("Lignes multiples", ["No", "Yes"])
        internet = select("Internet", ["Fiber optic", "DSL", "No"])
        addons = {col: select(label, ["No", "Yes"]) for col, label in ADDONS.items()}
    with c3:
        st.subheader("Contrat et facturation")
        contract = select("Contrat", ["Month-to-month", "One year", "Two year"])
        paperless = select("Facture dématérialisée", ["Yes", "No"])
        payment = select("Moyen de paiement", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = st.number_input("Mensualité ($)", min_value=0.0, max_value=200.0, value=70.0, step=5.0)
        total = st.number_input("Total facturé ($) - 0 = ancienneté x mensualité", min_value=0.0, max_value=10000.0, value=0.0, step=50.0)
    submitted = st.form_submit_button("Prédire", type="primary")

if submitted:
    if phone == "No":
        multiple = "No phone service"
    if internet == "No":
        addons = {col: "No internet service" for col in addons}

    row = pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": int(senior),
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone,
                "MultipleLines": multiple,
                "InternetService": internet,
                **addons,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly,
                "TotalCharges": total or tenure * monthly,
            }
        ]
    )
    X = prepare(row)
    cluster = int(cluster_model.predict(X)[0])
    proba = float(churn_model.predict_proba(X)[0, 1])
    churn = proba >= info["threshold"]
    segment = info["clusters"][str(cluster)]

    st.divider()
    k1, k2, k3 = st.columns(3)
    k1.metric("Profil", f"Cluster {cluster}")
    k2.metric("Probabilité de churn", f"{proba:.0%}")
    k3.metric("Prédiction", "Churn" if churn else "Reste")
    st.progress(proba)

    if churn:
        st.error(f"Client à risque : probabilité {proba:.0%} au-dessus du seuil {info['threshold']}.")
    else:
        st.success(f"Client plutôt fidèle : probabilité {proba:.0%} sous le seuil {info['threshold']}.")

    st.subheader(f"Segment : {segment['name']}")
    st.write(
        f"{segment['n']} clients d'entraînement · ancienneté moyenne {segment['tenure']:.0f} mois · "
        f"mensualité moyenne {segment['MonthlyCharges']:.0f} $ · {segment['nb_services']:.1f} services · "
        f"taux de churn du segment **{segment['churn_rate']:.0%}**"
    )
