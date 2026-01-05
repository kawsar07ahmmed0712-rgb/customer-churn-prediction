import argparse
import os
import logging

import joblib
import numpy as np
import pandas as pd

from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

TARGET_COL = "Churn"
DROP_COLS = ["customerID"]


def clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Strip whitespace in object columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # TotalCharges often contains blanks => convert to numeric (NaN if invalid)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop ID cols if present
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        # with_mean=False makes it compatible if we end up sparse
        ("scaler", StandardScaler(with_mean=False)),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    # sparse_threshold controls whether output becomes sparse.
    # Setting low value (0.0) encourages sparse output if any transformer is sparse.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.0
    )

    return preprocessor


def ensure_sparse(X):
    """
    Ensure X is a scipy sparse matrix so we can save using sparse.save_npz.
    If X is dense (numpy array), convert to CSR sparse.
    """
    if sparse.issparse(X):
        return X
    return sparse.csr_matrix(X)


def main(raw_path: str, out_dir: str, test_size: float, random_state: int):
    logging.info(f"Reading raw data: {raw_path}")
    df = pd.read_csv(raw_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset columns.")

    df = clean_telco(df)

    # Map target to 0/1
    y = df[TARGET_COL].map({"Yes": 1, "No": 0})
    if y.isna().any():
        bad_vals = df.loc[y.isna(), TARGET_COL].unique()
        raise ValueError(f"Unexpected values in '{TARGET_COL}': {bad_vals}. Expected Yes/No.")

    X = df.drop(columns=[TARGET_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y.values,
        test_size=test_size,
        random_state=random_state,
        stratify=y.values
    )

    logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    preprocessor = build_preprocessor(X_train)

    logging.info("Fitting preprocessor and transforming train/test...")
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # ✅ critical fix: ensure sparse before saving as npz
    X_train_t = ensure_sparse(X_train_t)
    X_test_t = ensure_sparse(X_test_t)

    os.makedirs(out_dir, exist_ok=True)

    # Save artifacts
    preprocessor_path = os.path.join(out_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)

    sparse.save_npz(os.path.join(out_dir, "X_train.npz"), X_train_t)
    sparse.save_npz(os.path.join(out_dir, "X_test.npz"), X_test_t)

    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test)

    logging.info(f"✅ Saved preprocessor: {preprocessor_path}")
    logging.info(f"✅ Saved processed arrays to: {out_dir}")
    logging.info(f"✅ Transformed train matrix type: {type(X_train_t)}, shape: {X_train_t.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Telco Churn dataset")
    parser.add_argument("--raw_path", type=str, default="data/raw/telco_churn.csv")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    main(
        raw_path=args.raw_path,
        out_dir=args.out_dir,
        test_size=args.test_size,
        random_state=args.random_state
    )
