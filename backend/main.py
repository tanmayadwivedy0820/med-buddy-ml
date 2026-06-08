import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.predictor import predict, model

load_dotenv()

# Resolve dataset path once, using the SAME env-var pattern as training/predictor
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve()
DATASET_PATH = PROJECT_ROOT / os.getenv("DATASET_DIR") / os.getenv("DATASET_NAME")
MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / os.getenv("MODEL_NAME")
TARGET_COL = os.getenv("TARGET_COL")


app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0"
)

# input schema matching training features
class HeartDiseaseInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


@app.get("/health")
def health_check():
    return {"status": "ok"}


# Heart disease prediction endpoint
@app.post("/predict-heart-disease")
def predict_heart_disease(input_data: HeartDiseaseInput):
    input_data = input_data.model_dump()
    result = predict(input_data=input_data)
    return {
        "prediction": result["prediction"],
        "probability": result["probability"],
        "diagnosis": (
            "Heart Disease Detected"
            if result["prediction"] == 1
            else "No Heart Disease Detected"
        )
    }

@app.get("/dataset-stats")
def dataset_stats():
    """
    Read-only EDA endpoint.
    Returns summary statistics, target class balance, and per-feature
    distributions so the frontend can render charts without ever
    touching the raw CSV or the model artifact directly.
    """
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Dataset file not found on server.")

    feature_cols = [c for c in df.columns if c != TARGET_COL]

    # Target balance (e.g. {0: 138, 1: 165})
    target_counts = df[TARGET_COL].value_counts().sort_index()
    target_balance = {str(k): int(v) for k, v in target_counts.items()}

    # Per-feature summary stats (count, mean, std, min, quartiles, max)
    describe = df[feature_cols].describe().round(3)
    summary = {col: describe[col].to_dict() for col in feature_cols}

    # Histograms: 20-bin counts per feature, for distribution charts
    # Histograms: 20-bin counts per feature, for distribution charts
    histograms = {}
    for col in feature_cols:
        binned = df[col].value_counts(bins=20, sort=False)
        histograms[col] = {
            "bin_edges": [round(float(interval.left), 3) for interval in binned.index]
                         + [round(float(binned.index[-1].right), 3)],
            "counts": [int(c) for c in binned.values],
        }

    return {
        "n_rows": int(df.shape[0]),
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "target_balance": target_balance,
        "summary": summary,
        "histograms": histograms,
    }


@app.get("/model-info")
def model_info():
    """
    Read-only model interpretability endpoint.
    Pulls feature importances out of the RandomForest step inside the
    saved Pipeline and returns them ranked, so the frontend can render
    an explainability chart without ever loading the model artifact.
    """
    # The saved artifact is a Pipeline: steps are 'scaler' then 'model'.
    # Importances live on the RandomForest, not the Pipeline itself.
    rf = model.named_steps["model"]

    # Feature order MUST match training order: CSV columns minus target.
    df = pd.read_csv(DATASET_PATH, nrows=0)  # header only — no data needed
    feature_names = [c for c in df.columns if c != TARGET_COL]

    importances = rf.feature_importances_

    # Pair names with importances and rank descending
    ranked = sorted(
        zip(feature_names, importances),
        key=lambda pair: pair[1],
        reverse=True,
    )

    return {
        "model_type": type(rf).__name__,
        "n_features": len(feature_names),
        "n_estimators": rf.n_estimators,
        "max_depth": rf.max_depth,
        "feature_importances": [
            {"feature": name, "importance": round(float(imp), 5)}
            for name, imp in ranked
        ],
    }