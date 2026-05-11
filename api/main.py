import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from pathlib import Path

app = FastAPI(
    title="Clinical Intelligence API",
    description="API for disease prediction, monitoring, and explainability",
    version="1.0.0"
)

# Paths
API_DIR = Path(__file__).parent.resolve()
ROOT_DIR = API_DIR.parent
MODELS_DIR = ROOT_DIR / "models" / "saved"

# Global model cache to hold loaded models
MODEL_CACHE = {}

# --- Pydantic Models ---

class PredictRequest(BaseModel):
    disease: str
    features: Dict[str, Any]

# --- Helper Functions ---

def get_model_path(disease: str, version: str = "best"):
    """Get the path for the requested model."""
    if version == "best":
        return MODELS_DIR / f"{disease}_best_model.pkl"
    return MODELS_DIR / f"{disease}_best_model_{version}.pkl"

def load_model(disease: str):
    """Load model into cache if not present."""
    if disease not in MODEL_CACHE:
        model_path = get_model_path(disease)
        if not model_path.exists():
            return None
        MODEL_CACHE[disease] = joblib.load(model_path)
    return MODEL_CACHE[disease]

# --- Endpoints ---

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Clinical API is running."}

@app.post("/predict")
def predict(payload: PredictRequest):
    """Predict using the requested disease model."""
    disease = payload.disease.lower()
    features = payload.features
    
    model = load_model(disease)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model for '{disease}' not found.")
        
    try:
        # Load artifacts for scaler and feature order
        artifact_path = MODELS_DIR / f"{disease}_artifacts.pkl"
        if not artifact_path.exists():
            raise HTTPException(status_code=404, detail=f"Artifacts for '{disease}' not found.")
            
        art = joblib.load(artifact_path)
        feature_names = art.get("feature_names")
        scaler = art.get("scaler")
        
        # Ensure we have a DataFrame with exactly the right columns
        if feature_names:
            # Filter and order features
            ordered_features = {k: features.get(k, 0) for k in feature_names}
            df = pd.DataFrame([ordered_features])
        else:
            df = pd.DataFrame([features])
            
        # Scale only the columns that the scaler was fitted on
        if scaler and hasattr(scaler, "feature_names_in_"):
            cols_to_scale = list(scaler.feature_names_in_)
            df[cols_to_scale] = scaler.transform(df[cols_to_scale])
            X_input = df.values
        else:
            X_input = df.values
            
        prediction = model.predict(X_input)[0]
        probability = model.predict_proba(X_input)[0][1] if hasattr(model, "predict_proba") else None
        
        return {
            "disease": disease,
            "prediction": int(prediction),
            "probability": float(probability) if probability is not None else None,
            "risk_level": "High" if int(prediction) == 1 else "Low"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/update_model")
def update_model(payload: dict):
    """Endpoint to update a model (simulated)."""
    disease = payload.get("disease", "").lower()
    if not disease:
        raise HTTPException(status_code=400, detail="Require 'disease'.")
    
    if disease in MODEL_CACHE:
        del MODEL_CACHE[disease]
        
    return {"status": "success", "message": f"Model for {disease} flagged for reload."}

@app.post("/rollback")
def rollback(payload: dict):
    """Endpoint to rollback a model (simulated)."""
    disease = payload.get("disease", "").lower()
    version = payload.get("version", "previous")
    if not disease:
        raise HTTPException(status_code=400, detail="Require 'disease'.")
        
    prev_model_path = get_model_path(disease, version)
    if not prev_model_path.exists():
        raise HTTPException(status_code=404, detail=f"Rollback version '{version}' for '{disease}' not found.")
        
    MODEL_CACHE[disease] = joblib.load(prev_model_path)
    return {"status": "success", "message": f"Rolled back {disease} to version '{version}'."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
