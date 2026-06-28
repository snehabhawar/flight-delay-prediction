import pandas as pd
import glob

# Columns we will NEVER use as features.
# Each is excluded for a specific, defensible reason:
LEAKAGE_AND_LABEL = [
    "ARR_DEL15",   # the LABEL — what we predict, not an input
    "ARR_DELAY",   # leakage: known only AFTER landing (label is derived from it)
    "DEP_DELAY",   # leakage: known only AFTER departure = after prediction time
    "CANCELLED",   # used to filter rows, not a feature (unknown in advance)
]

def load_data(raw_dir="data/raw"):
    """Load all monthly BTS CSVs, stack them, drop unusable rows,
    and split into features (X) and label (y)."""

    # 1. Find every CSV in the raw folder and stack them into one DataFrame.
    #    glob finds all matching files so we don't hardcode three filenames.
    files = glob.glob(f"{raw_dir}/*.csv")
    print(f"Found {len(files)} files: {files}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"Combined shape: {df.shape}")

    # 2. Drop rows we can't learn from: cancelled / diverted flights have
    #    no arrival, so ARR_DEL15 is NaN. A flight that didn't arrive has
    #    no "was it delayed" answer.
    before = len(df)
    df = df.dropna(subset=["ARR_DEL15"])
    print(f"Dropped {before - len(df)} rows with missing label (cancelled/diverted)")

    # 3. Separate the label (y) from the features (X).
    y = df["ARR_DEL15"].astype(int)          # 0/1 target
    X = df.drop(columns=LEAKAGE_AND_LABEL)   # everything except label + leakage

    print(f"Feature columns kept: {X.columns.tolist()}")
    print(f"X shape: {X.shape},  y shape: {y.shape}")
    print(f"Class balance (delayed fraction): {y.mean():.3f}")

    return X, y


# Lets us run `python src/data.py` directly to sanity-check.
if __name__ == "__main__":
    X, y = load_data()