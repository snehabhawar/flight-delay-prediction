# Flight Delay Prediction

Predicting US domestic flight delays (≥15 min, per BTS definition) from
Bureau of Transportation Statistics on-time performance data.

> **Resume line — NOT YET EARNED. Do not claim until the code does it.**
> "Built and trained a feedforward neural network in PyTorch to predict US
> domestic flight delays on 1M+ BTS records (AUC X.XX), with feature
> engineering across carrier, route, time, and weather signals."

## Status
- [ ] Step 1 — baseline (pandas + sklearn logistic regression / gradient boosting)
- [ ] Step 2 — same model rebuilt as logistic regression in PyTorch
- [ ] Step 3 — feedforward neural net (hidden layers, ReLU, dropout, DataLoader)
- [ ] Step 4 — polish: train/val/test split, confusion matrix, ROC curve, this README

## Data
US DOT Bureau of Transportation Statistics — "Reporting Carrier On-Time
Performance." Public. CSVs live in `data/raw/` (gitignored, not committed).
Target label: `ArrDel15` (1 if arrival delay ≥ 15 min).

### Leakage note (the important one)
Columns known only *after* a flight lands are excluded from features:
`ArrDelay`, `DepDelay`, `ActualDepTime`, all `*DelayCause` columns. They are
downloaded but quarantined — using them would leak the answer.

## Features (filled in as engineered)
_TBD — carrier, origin/dest, distance, day-of-week, scheduled dep time bucket._

## Results (filled in as earned)
_TBD — baseline AUC vs. neural net AUC._

## Repo layout
```
data/raw/         BTS CSVs (gitignored)
data/processed/   cleaned/feature parquet (gitignored)
src/              pipeline + model code
notebooks/        exploration
models/           saved weights (gitignored)
reports/          confusion matrix, ROC curve
```
