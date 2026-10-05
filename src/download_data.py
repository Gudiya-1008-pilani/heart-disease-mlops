from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


# Find the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define where the raw dataset will be saved
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_FILE = RAW_DATA_DIR / "heart_disease_uci.csv"


def download_data():
    """Download the UCI Heart Disease dataset and save it as a CSV file."""

    print("Downloading Heart Disease UCI dataset...")

    # Dataset ID 45 corresponds to the UCI Heart Disease dataset
    heart_disease = fetch_ucirepo(id=45)

    # Separate features and target returned by UCI
    features = heart_disease.data.features
    targets = heart_disease.data.targets

    # Combine features and target into one DataFrame
    dataframe = pd.concat([features, targets], axis=1)

    # Make sure the destination directory exists
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save the raw dataset
    dataframe.to_csv(RAW_DATA_FILE, index=False)

    print("Dataset downloaded successfully.")
    print(f"Dataset shape: {dataframe.shape}")
    print(f"Dataset saved to: {RAW_DATA_FILE}")


if __name__ == "__main__":
    download_data()