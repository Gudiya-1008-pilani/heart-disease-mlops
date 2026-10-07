import pandas as pd


def test_processed_dataset_exists():
    df = pd.read_csv("data/processed/heart_disease_processed.csv")

    assert not df.empty


def test_target_is_binary():
    df = pd.read_csv("data/processed/heart_disease_processed.csv")

    assert set(df["target"].unique()).issubset({0, 1})


def test_expected_columns_exist():
    df = pd.read_csv("data/processed/heart_disease_processed.csv")

    expected_columns = {
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target",
    }

    assert set(df.columns) == expected_columns