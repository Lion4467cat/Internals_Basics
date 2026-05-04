from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
import json
import os

app = FastAPI(title="ShieldOps Resolution Hours API")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, "models", "best_model.pkl"))


class PredictRequest(BaseModel):
    severity_level: int = Field(..., ge=1, le=5)
    alerts_count: int = Field(..., ge=1, le=50)
    analyst_experience: int = Field(..., ge=1, le=15)
    is_automated: int = Field(..., ge=0, le=1)


@app.get("/health")
def health():
    return {"alive": True, "service": "ShieldOps resolution_hours API"}


@app.post("/predict")
def predict(req: PredictRequest):
    features = np.array(
        [
            [
                req.severity_level,
                req.alerts_count,
                req.analyst_experience,
                req.is_automated,
            ]
        ]
    )
    prediction = model.predict(features)[0]

    # Save results
    os.makedirs("results", exist_ok=True)
    output = {
        "health_endpoint": "/health",
        "predict_endpoint": "/predict",
        "port": 9000,
        "health_response": {"alive": True, "service": "ShieldOps resolution_hours API"},
        "test_input": {
            "severity_level": req.severity_level,
            "alerts_count": req.alerts_count,
            "analyst_experience": req.analyst_experience,
            "is_automated": req.is_automated,
        },
        "prediction": round(float(prediction), 4),
    }
    with open("results/step3_s4.json", "w") as f:
        json.dump(output, f, indent=2)

    return {"prediction": round(float(prediction), 4)}
