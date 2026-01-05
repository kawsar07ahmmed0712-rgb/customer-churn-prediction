import os
import pandas as pd
from sklearn.datasets import fetch_openml

RAW_DATA_PATH = "data/raw/telco_churn.csv"

def download_telco_churn():
    print("✅ Downloading Telco Customer Churn dataset from OpenML...")

    data = fetch_openml(data_id=42178, as_frame=True)
    df = data.frame

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)

    print(f"✅ Dataset saved successfully at: {RAW_DATA_PATH}")
    print(f"✅ Dataset shape: {df.shape}")

if __name__ == "__main__":
    download_telco_churn()
