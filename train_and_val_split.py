import pandas as pd
import numpy as np

# ==========================
# Configuration
# ==========================

INPUT_FILE = "train.txt"   # archivo original
TRAIN_FILE = "train70.txt"
VAL_FILE = "val30.txt"

VAL_RATIO = 0.30
RANDOM_SEED = 42

# ==========================
# Load data
# ==========================

df = pd.read_csv(
    INPUT_FILE,
    sep=r"\s+",
    header=None
)

# First column = unit ID
unit_col = 0

# ==========================
# Split by units
# ==========================

units = df[unit_col].unique()

rng = np.random.default_rng(RANDOM_SEED)

n_val_units = int(round(len(units) * VAL_RATIO))

val_units = rng.choice(
    units,
    size=n_val_units,
    replace=False
)

val_units = set(val_units)

# Training units = all others
train_units = [u for u in units if u not in val_units]

# ==========================
# Create datasets
# ==========================

train_df = df[df[unit_col].isin(train_units)].copy()
val_df = df[df[unit_col].isin(val_units)].copy()

# ==========================
# Save files
# ==========================

train_df.to_csv(
    TRAIN_FILE,
    sep=" ",
    header=False,
    index=False
)

val_df.to_csv(
    VAL_FILE,
    sep=" ",
    header=False,
    index=False
)

# ==========================
# Summary
# ==========================

print(f"Total units:      {len(units)}")
print(f"Training units:   {len(train_units)}")
print(f"Validation units: {len(val_units)}")

print()
print(f"Training rows:    {len(train_df)}")
print(f"Validation rows:  {len(val_df)}")

print()
print(f"Saved: {TRAIN_FILE}")
print(f"Saved: {VAL_FILE}")