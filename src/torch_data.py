import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data import load_data

def prepare_torch_data():
    """Load data and return PyTorch tensors, with categoricals as integer IDs
    (ready for embeddings later) and numerics scaled."""

    X, y = load_data()

    categorical = ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "DEP_TIME_BLK"]
    numeric = ["MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "CRS_DEP_TIME", "DISTANCE"]

    # --- Integer-encode each categorical column ---
    # Map each distinct value (e.g. each airport) to an integer ID: ATL->0, ORD->1...
    # We also record how many distinct values each column has (the "cardinality"),
    # because the embedding layer in Step 3 needs to know how big each lookup table is.
    cardinalities = {}
    for col in categorical:
        X[col] = X[col].astype("category")        # pandas assigns a category code per value
        cardinalities[col] = X[col].cat.categories.size  # how many distinct values
        X[col] = X[col].cat.codes.astype("int64") # replace text with its integer code

    print("Cardinalities (distinct values per categorical):", cardinalities)

    # --- Split BEFORE scaling (same leakage-safe principle as baseline) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Scale numerics: fit on train only, apply to both ---
    scaler = StandardScaler()
    X_train[numeric] = scaler.fit_transform(X_train[numeric])
    X_test[numeric] = scaler.transform(X_test[numeric])

    # --- Convert to PyTorch tensors ---
    # Categoricals stay as integers (int64) — they're IDs, not quantities.
    # Numerics become floats. Label becomes float for BCE loss.
    X_train_cat = torch.tensor(X_train[categorical].values, dtype=torch.int64)
    X_test_cat  = torch.tensor(X_test[categorical].values,  dtype=torch.int64)
    X_train_num = torch.tensor(X_train[numeric].values, dtype=torch.float32)
    X_test_num  = torch.tensor(X_test[numeric].values,  dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test.values,  dtype=torch.float32)

    print(f"Train categoricals: {X_train_cat.shape}, numerics: {X_train_num.shape}")
    print(f"Test  categoricals: {X_test_cat.shape}, numerics: {X_test_num.shape}")

    return (X_train_cat, X_train_num, y_train_t,
            X_test_cat,  X_test_num,  y_test_t,
            cardinalities)


if __name__ == "__main__":
    prepare_torch_data()