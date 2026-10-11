# Analyse, segmentation et prédiction du churn client

Solution de Machine Learning sur le dataset **Telco Customer Churn** (7 043 clients) :
EDA → segmentation K-Means → classification du churn → optimisation → suivi MLflow → application Streamlit conteneurisée.

## Structure

```
data/raw/data.csv                  dataset brut
data/processed/clean_eng.csv       données nettoyées + feature engineering (notebook 01)
notebooks/01_eda.ipynb             analyse exploratoire, valeurs manquantes, outliers, corrélations
notebooks/02_clusters.ipynb        K-Means (Elbow, Silhouette, DB, CH), Agglomerative, DBSCAN, PCA, profils
notebooks/03_classification.ipynb  6 modèles, déséquilibre (class_weight, oversampling, SMOTE)
notebooks/04_lien.ipynb            apport du cluster et du feature engineering
notebooks/05_optimisation_mlflow.ipynb  GridSearch / RandomizedSearch, modèle final, comparaison MLflow
artifacts/                         modèle final, modèle de clustering, model_info.json (utilisés par l'app)
reports/figures/                   graphiques
reports/mlflow_runs.csv            export de toutes les expériences MLflow
app/streamlit_app.py               application de prédiction
docker/Dockerfile                  image de l'application
```

## Installation

```bash
pip install -r requirements.txt
```

Exécuter les notebooks **dans l'ordre** 01 → 05. Chaque notebook contient tout son code (aucun module externe).
01 produit `data/processed/clean_eng.csv` utilisé par les suivants, et 05 lit les runs MLflow de 03.

## Choix principaux

- **Données :** 11 `TotalCharges` vides (clients à `tenure = 0`) remplacés par 0. Aucun doublon, aucun outlier IQR.
  Le dataset ne contient pas les colonnes âge / satisfaction / réclamations citées dans le brief.
- **Feature engineering :** `charges_per_tenure`, `is_new_customer`, `is_month_to_month`, `is_fiber`, `is_electronic_check`, `nb_services`, `tenure_band`.
- **Clustering :** K-Means k = 3, sans la variable `Churn`, et sans `TotalCharges` (corrélée à 0.83 avec `tenure`).
  Trois profils : *nouveaux clients à risque* (~49 % de churn), *fidèles premium* (~15 %), *basiques sans internet* (~7 %).
- **Anti data leakage :** split train/test 80/20 stratifié en premier. Imputation, scaling, encodage, K-Means (variable `cluster`)
  et SMOTE sont des étapes d'un `imblearn.Pipeline` : ils sont ajustés sur le train uniquement, et dans chaque fold de CV.
  Le seuil de décision est choisi sur des prédictions out-of-fold du train. Le test n'est utilisé que pour le score final.
- **Évaluation :** Precision, Recall, F1, ROC-AUC (CV 5 folds + test), matrices de confusion.

## Résultats

| Étape | Constat |
|---|---|
| Déséquilibre | SMOTE / class_weight : recall de ~53 % à ~80 % (Régression Logistique) |
| Cluster comme variable | gain de ROC-AUC entre -0.006 et +0.001 : non significatif (écart-type CV ≈ 0.011) |
| Feature engineering | léger gain pour les modèles simples (LogReg +0.002, RF +0.004), dans le bruit |
| Optimisation | Random Forest et XGBoost +0.025 de ROC-AUC, LogReg et GB déjà à l'optimum |

**Modèle final :** Régression Logistique (C = 0.1) + SMOTE + feature engineering, seuil 0.56.

| Test | ROC-AUC | Recall | Precision | F1 |
|---|---|---|---|---|
| Modèle final | 0.844 | 0.75 | 0.54 | 0.63 |

Le cluster n'améliore pas la prédiction : il n'est pas utilisé comme entrée du modèle final. Il sert à **l'interprétation** et il est affiché dans l'application.

## MLflow

Chaque expérience (notebooks 03, 04, 05) enregistre : hyperparamètres, paramètres de preprocessing (FE, cluster, sampler,
scaler, encodage, imputation, seuil), métriques CV et test, matrice de confusion, courbe ROC et modèle.
Le modèle final est tagué `final` et enregistré dans le registre sous `churn-model`.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Puis ouvrir http://localhost:5000 (expérience `churn-prediction`). Un export est disponible dans `reports/mlflow_runs.csv`.

## Application Streamlit

```bash
streamlit run app/streamlit_app.py
```

L'utilisateur saisit les caractéristiques d'un client. L'application affiche son **cluster / profil**, la **probabilité de churn** et la **prédiction** (selon le seuil du modèle).

## Docker

```bash
docker build -f docker/Dockerfile -t churn-app .
```

```bash
docker run -p 8501:8501 churn-app
```

Puis ouvrir http://localhost:8501.
