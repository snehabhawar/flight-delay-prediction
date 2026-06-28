import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from data import load_data

X, y = load_data()

categorical = ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "DEP_TIME_BLK"]
numeric = ["MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "CRS_DEP_TIME", "DISTANCE"]

# Same split as baseline (random_state=42) so the comparison is controlled:
# same rows, same encoding, ONLY the model differs.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape},  Test: {X_test.shape}")

# One-hot the categoricals. Tried native categorical handling first, but
# HistGradientBoosting caps native categoricals at 255 distinct values and
# ORIGIN/DEST have ~348 airports each — so one-hot is the right tool here.
# sparse_output=False because HistGradientBoosting requires dense input.
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("num", StandardScaler(), numeric),
    ]
)

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", HistGradientBoostingClassifier(random_state=42)),
])

print("Training gradient boosting...")
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print(f"\nGradient Boosting — Test AUC: {auc:.4f}")
print(f"(Baseline logistic regression was 0.6659)")