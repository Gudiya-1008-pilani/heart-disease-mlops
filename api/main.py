import logging
import time

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel
from starlette.responses import Response


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

MODEL_PATH = "models/heart_disease_pipeline.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0",
)


REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status"],
)

PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Number of model predictions",
    ["prediction"],
)

PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Model prediction latency",
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


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    response = await call_next(request)

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    ).inc()

    return response


@app.get("/")
def root():
    return {
        "message": "Heart Disease Prediction API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type="text/plain",
    )


@app.post("/predict")
def predict(patient: PatientInput):
    start_time = time.perf_counter()

    input_data = pd.DataFrame([patient.model_dump()])

    prediction = int(model.predict(input_data)[0])

    probabilities = model.predict_proba(input_data)[0]

    confidence = float(max(probabilities))

    latency = time.perf_counter() - start_time

    PREDICTION_LATENCY.observe(latency)

    PREDICTION_COUNT.labels(
        prediction=str(prediction)
    ).inc()

    logger.info(
        "prediction=%s confidence=%.4f latency=%.4f",
        prediction,
        confidence,
        latency,
    )

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "latency_seconds": round(latency, 4),
    }