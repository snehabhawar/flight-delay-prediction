import pandas as pd

df = pd.read_csv("data/raw/T_ONTIME_REPORTING.csv")

print("Shape (rows, columns):", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 3 rows:")
print(df.head(3))

# Real BTS column name is ARR_DEL15 (uppercase, underscores)
print("\nARR_DEL15 value counts (raw, including NaN):")
print(df["ARR_DEL15"].value_counts(dropna=False))

print("\nARR_DEL15 as proportions (including NaN):")
print(df["ARR_DEL15"].value_counts(normalize=True, dropna=False))

# How many cancelled flights? (these are the NaN labels)
print("\nCANCELLED value counts:")
print(df["CANCELLED"].value_counts(dropna=False))