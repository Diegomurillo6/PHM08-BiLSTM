import pickle
import joblib
import numpy as np
import pandas as pd

# =====================================================
# PARAMETERS
# =====================================================

MIN_WINDOW_LENGTH = 30

K = 0.001
B = 0.95

# =====================================================
# LOAD FILES
# =====================================================

val30_df = pd.read_csv(
    "val30.txt",
    sep=r"\s+",
    header=None
)

cluster_norm = pd.read_csv(
    "cluster_normalization.csv"
)

offline_hi = pd.read_csv(
    "offline_hi_library.csv"
)

lr_model = joblib.load(
    "HI_linear_regression.pkl"
)

kmeans = joblib.load(
    "operating_condition_kmeans.pkl"
)

# =====================================================
# COLUMN DEFINITIONS
# =====================================================

UNIT_COL = 0
CYCLE_COL = 1

OP_COLS = [2, 3, 4]

sensor_cols = [
    6,   # sensor 2
    7,   # sensor 3
    8,   # sensor 4
    11,  # sensor 7
    12,  # sensor 8
    13,  # sensor 9
    15,  # sensor 11
    16,  # sensor 12
    17,  # sensor 13
    18,  # sensor 14
    19,  # sensor 15
    21,  # sensor 17
    24,  # sensor 20
    25   # sensor 21
]

for sensor in sensor_cols:
    val30_df[sensor] = val30_df[sensor].astype(float)

# =====================================================
# NORMALIZE VALIDATION DATASET
# =====================================================

print("Predicting operating-condition clusters...")

clusters = kmeans.predict(
    val30_df[OP_COLS].values
)

val30_df["cluster"] = clusters

print("Normalizing sensors...")

for sensor in sensor_cols:

    for cluster in val30_df["cluster"].unique():

        mask = val30_df["cluster"] == cluster

        row = cluster_norm[
            (cluster_norm["cluster"] == cluster)
            &
            (cluster_norm["sensor"] == sensor)
        ]

        if len(row) == 0:
            continue

        mean = row["mean"].iloc[0]
        std = row["std"].iloc[0]

        if std == 0:
            continue

        val30_df.loc[mask, sensor] = (
            val30_df.loc[mask, sensor] - mean
        ) / std

print("Normalization complete")

# =====================================================
# BUILD HI LIBRARY
# =====================================================

hi_library = {}

for unit in offline_hi["unit"].unique():

    curve = (
        offline_hi[
            offline_hi["unit"] == unit
        ]
        .sort_values("cycle")
        ["HI_fitted"]
        .values
    )

    hi_library[unit] = curve

print(
    f"Loaded {len(hi_library)} HI curves"
)

# =====================================================
# GENERATE ONE RANDOM WINDOW
# PER VALIDATION UNIT
# =====================================================

validation_segments = []

for unit in val30_df[UNIT_COL].unique():

    unit_df = (
        val30_df[
            val30_df[UNIT_COL] == unit
        ]
        .sort_values(CYCLE_COL)
        .reset_index(drop=True)
    )

    n = len(unit_df)

    if n <= MIN_WINDOW_LENGTH:
        continue

    start = np.random.randint(
        0,
        n - MIN_WINDOW_LENGTH
    )

    end = np.random.randint(
        start + MIN_WINDOW_LENGTH,
        n
    )

    segment = unit_df.iloc[start:end + 1]

    true_rul = n - end - 1

    validation_segments.append({

    "unit":
        unit,

    "segment":
        segment,

    "true_rul":
        true_rul,

    "start_cycle":
        int(
            segment.iloc[0][CYCLE_COL]
        ),

    "end_cycle":
        int(
            segment.iloc[-1][CYCLE_COL]
        )
})

print(
    f"Created {len(validation_segments)} validation windows"
)

# =====================================================
# HI PREDICTION
# =====================================================

def predict_hi_curve(df):

    X = df[sensor_cols].astype(float).values

    hi = lr_model.predict(X)

    return np.asarray(hi).flatten()

# =====================================================
# SECTION 2.3
# SIMILARITY MATCHING
# =====================================================

def estimate_rul(test_hi):

    T0 = len(test_hi)

    best_similarity = -np.inf

    best_unit = None
    best_shift = None
    best_segment = None
    best_train_curve = None

    similarities = []
    rul_candidates = []

    for train_unit, train_hi in hi_library.items():

        Tj = len(train_hi)

        if Tj <= T0:
            continue

        for tau in range(Tj - T0 + 1):

            train_segment = train_hi[
                tau:tau+T0
            ]

            distance = np.mean(
                (test_hi - train_segment)**2
            )

            similarity = np.exp(
                -distance / K
            )

            rul = (
                Tj
                - T0
                - tau
            )

            similarities.append(similarity)
            rul_candidates.append(rul)

            if similarity > best_similarity:

                best_similarity = similarity

                best_unit = train_unit

                best_shift = tau

                best_segment = train_segment.copy()

                best_train_curve = train_hi.copy()

    similarities = np.array(similarities)
    rul_candidates = np.array(rul_candidates)

    threshold = (
        B *
        similarities.max()
    )

    selected = (
        similarities >= threshold
    )

    estimated_rul = np.sum(
        similarities[selected]
        *
        rul_candidates[selected]
    ) / np.sum(
        similarities[selected]
    )

    return {
        "predicted_rul": estimated_rul,
        "best_unit": best_unit,
        "tau": best_shift,
        "best_similarity": best_similarity,
        "matched_segment": best_segment,
        "matched_curve": best_train_curve
    }

# =====================================================
# PREDICT VALIDATION SET
# =====================================================

prediction_details = []

y_true = []
y_pred = []

for sample in validation_segments:

    unit = sample["unit"]

    true_rul = sample["true_rul"]

    start_cycle = sample["start_cycle"]

    end_cycle = sample["end_cycle"]

    hi_curve = predict_hi_curve(
        sample["segment"]
    )

    result = estimate_rul(
        hi_curve
    )

    predicted_rul = result[
        "predicted_rul"
    ]

    y_true.append(true_rul)

    y_pred.append(predicted_rul)

    prediction_details.append({

        "validation_unit":
            unit,

        "window_start":
            start_cycle,

        "window_end":
            end_cycle,

        "window_length":
            len(hi_curve),

        "actual_rul":
            true_rul,

        "predicted_rul":
            predicted_rul,

        "validation_hi":
            hi_curve,

        "matched_unit":
            result["best_unit"],

        "tau":
            result["tau"],

        "similarity":
            result["best_similarity"],

        "matched_segment":
            result["matched_segment"],

        "matched_curve":
            result["matched_curve"],

        "matched_curve_length":
            len(
                result["matched_curve"]
            ),

        "predicted_failure_cycle":
            len(
                result["matched_curve"]
            ) - 1
    })

y_true = np.array(y_true)
y_pred = np.array(y_pred)


# Save windows and data
with open(
    "validation_debug_data.pkl",
    "wb"
) as f:

    pickle.dump(
        prediction_details,
        f
    )

# =====================================================
# METRICS (SECTION 3.1)
# =====================================================

errors = y_pred - y_true

# RMSE

rmse = np.sqrt(
    np.mean(errors**2)
)

# Accuracy

accuracy = (
    np.sum(
        (errors >= -13)
        &
        (errors <= 10)
    )
    / len(errors)
) * 100

# Score

score = 0

for e in errors:

    if e < 0:

        score += (
            np.exp(
                -e / 13
            ) - 1
        )

    else:

        score += (
            np.exp(
                e / 10
            ) - 1
        )

# =====================================================
# RESULTS
# =====================================================

results = pd.DataFrame({
    "Actual_RUL": y_true,
    "Predicted_RUL": y_pred,
    "Error": errors
})

results.to_csv(
    "validation_results.csv",
    index=False
)

print()
print(f"Samples   : {len(y_true)}")
print(f"RMSE      : {rmse:.3f}")
print(f"Accuracy  : {accuracy:.2f}%")
print(f"Score     : {score:.3f}")
