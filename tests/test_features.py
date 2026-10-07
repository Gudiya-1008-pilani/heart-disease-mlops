import joblib
import pandas as pd


def test_saved_model_exists():
    model = joblib.load("models/heart_disease_pipeline.joblib")

    assert model is not None


def test_model_can_predict():
    model = joblib.load("models/heart_disease_pipeline.joblib")

    df = pd.read_csv("data/processed/heart_disease_processed.csv")
    X = df.drop(columns=["target"])

    predictions = model.predict(X.head(5))

    assert len(predictions) == 5
    assert set(predictions).issubset({0, 1})