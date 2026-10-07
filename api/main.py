import logging
import time

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

MODEL_PATH = "models/heart_disease_pipeline.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0"
)


class PatientInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float | None = None
    thal: float | None = None


@app.get("/")
def root():
    return {
        "message": "Heart Disease Prediction API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(patient: PatientInput):
    start_time = time.perf_counter()

    input_data = pd.DataFrame([patient.model_dump()])

    prediction = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]
    confidence = float(max(probabilities))

    elapsed_time = time.perf_counter() - start_time

    logger.info(
        "Prediction completed: prediction=%s confidence=%.4f latency=%.4fs",
        prediction,
        confidence,
        elapsed_time
    )

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4)
    }