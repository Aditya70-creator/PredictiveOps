import json
import os
import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data.generate_sensor_data import FEATURES
from src.monitoring.drift import evaluate_sensor_drift

app = FastAPI(
    title="PredictiveOps API",
    description="Predictive Maintenance & Industrial Anomaly Detection Platform",
    version="1.0.0",
)

MODEL_PATH = "models/predictive_pipeline.joblib"
META_PATH = "models/metadata.json"

class SensorInput(BaseModel):
    sensor_temperature_c: float = Field(..., ge=0, json_schema_extra={"example": 82.5})
    vibration_mm_s: float = Field(..., ge=0, json_schema_extra={"example": 3.8})
    pressure_bar: float = Field(..., ge=0, json_schema_extra={"example": 28.0})
    rotational_speed_rpm: float = Field(..., ge=0, json_schema_extra={"example": 1420.0})
    operating_hours: int = Field(..., ge=0, json_schema_extra={"example": 2400})
    torque_nm: float = Field(..., ge=0, json_schema_extra={"example": 58.5})

def compute_shap_values(classifier, X_trans: np.ndarray) -> np.ndarray:
    if hasattr(classifier, "get_booster"):
        dmat = xgb.DMatrix(X_trans)
        return classifier.get_booster().predict(dmat, pred_contribs=True)[0][:-1]
    elif hasattr(classifier, "coef_"):
        return (classifier.coef_[0] * X_trans[0]).astype(float)
    else:
        explainer = shap.TreeExplainer(classifier)
        sv = explainer.shap_values(X_trans)
        if isinstance(sv, list):
            return np.asarray(sv[1][0], dtype=float)
        elif len(sv.shape) == 3:
            return np.asarray(sv[0, :, 1], dtype=float)
        return np.asarray(sv[0], dtype=float)

def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
        raise HTTPException(status_code=503, detail="Model artifacts not found.")
    pipeline = joblib.load(MODEL_PATH)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    return pipeline, meta

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": os.path.exists(MODEL_PATH)}

@app.post("/predict")
def predict_failure(sensor: SensorInput):
    pipeline, meta = load_artifacts()
    df_in = pd.DataFrame([sensor.model_dump()])[FEATURES]

    prob = float(pipeline.predict_proba(df_in)[0, 1])
    threshold = float(meta["optimal_threshold"])
    is_imminent = bool(prob >= threshold)

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    X_trans = preprocessor.transform(df_in)

    vals = compute_shap_values(classifier, X_trans)
    contributions = {feat: round(float(val), 4) for feat, val in zip(FEATURES, vals)}
    top_drivers = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:3])

    return {
        "prediction": "FAILURE_IMMINENT" if is_imminent else "OPERATING_NORMALLY",
        "failure_probability": round(prob, 4),
        "decision_threshold": threshold,
        "winning_model": meta["winning_model"],
        "top_shap_drivers": top_drivers,
    }

@app.get("/drift-status")
def check_drift(use_drifted_batch: bool = True):
    ref_path = "data/raw/sensor_reference.csv"
    cur_path = "data/raw/sensor_drifted.csv" if use_drifted_batch else ref_path
    ref_df = pd.read_csv(ref_path)
    cur_df = pd.read_csv(cur_path)
    return evaluate_sensor_drift(ref_df, cur_df)