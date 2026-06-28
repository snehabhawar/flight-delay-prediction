import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from data import load_data   # our loader from src/data.py

# 1. Load the clean features and label
X, y = load_data()

# 2. Identify which columns are categorical (text) vs numeric.
categorical = ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "DEP_TIME_BLK"]
numeric = ["MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "CRS_DEP_TIME", "DISTANCE"]

# 3. Split into train/test BEFORE encoding. The test set is the honest exam.
#    stratify=y keeps the same delayed-fraction (~22%) in both halves.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape},  Test: {X_test.shape}")

# 4. Preprocessing: one-hot the categoricals, pass numerics through unchanged.
#    handle_unknown='ignore' = if a rare airport shows up only in test,
#    don't crash — just encode it as all-zeros.
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # add StandardScaler

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", StandardScaler(), numeric),   # was "passthrough" — now scales numerics
    ]
)

# 5. Chain preprocessing + model into one Pipeline.
#    max_iter raised because logistic regression needs more steps to
#    converge on this many one-hot columns.
model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", LogisticRegression(max_iter=1000)),
])

# 6. Fit on train ONLY. The pipeline learns the encoding from train,
#    then trains logistic regression on the encoded train data.
print("Training...")
model.fit(X_train, y_train)

# 7. Predict PROBABILITIES on the locked-away test set.
#    predict_proba gives [P(on-time), P(delayed)]; we want column 1 = P(delayed).
y_prob = model.predict_proba(X_test)[:, 1]

# 8. The number that matters: AUC on the test set.
auc = roc_auc_score(y_test, y_prob)
print(f"\nBaseline Logistic Regression — Test AUC: {auc:.4f}")