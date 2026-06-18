import pickle
import joblib
import numpy as np
import pandas as pd

# =====================================================
# PARAMETERS
# =====================================================

K = 0.001
B = 0.95

# =====================================================
# LOAD FILES
# =====================================================

# NOTA: si test.xlsx trae encabezados de columna, usar header=0
# en lugar de header=None.
test_df = pd.read_excel(
    "test.xlsx",
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
# (mismas posiciones que en val30.txt: unit, cycle,
# 3 condiciones operativas, 21 sensores)
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
    test_df[sensor] = test_df[sensor].astype(float)

# =====================================================
# NORMALIZE TEST DATASET
# =====================================================

print("Predicting operating-condition clusters...")

clusters = kmeans.predict(
    test_df[OP_COLS].values
)

test_df["cluster"] = clusters

print("Normalizing sensors...")

for sensor in sensor_cols:

    for cluster in test_df["cluster"].unique():

        mask = test_df["cluster"] == cluster

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

        test_df.loc[mask, sensor] = (
            test_df.loc[mask, sensor] - mean
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
# HI PREDICTION
# =====================================================

def predict_hi_curve(df):

    X = df[sensor_cols].astype(float).values

    hi = lr_model.predict(X)

    return np.asarray(hi).flatten()

# =====================================================
# SIMILARITY MATCHING
# (idéntico al usado en la validación)
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
# PREDICT TEST SET
# (se usa la trayectoria COMPLETA disponible por motor,
# sin recorte ni selección aleatoria de ventana)
# =====================================================

prediction_details = []
rul_table = []

units = sorted(
    test_df[UNIT_COL].unique()
)

for unit in units:

    unit_df = (
        test_df[
            test_df[UNIT_COL] == unit
        ]
        .sort_values(CYCLE_COL)
        .reset_index(drop=True)
    )

    hi_curve = predict_hi_curve(
        unit_df
    )

    result = estimate_rul(
        hi_curve
    )

    predicted_rul = result[
        "predicted_rul"
    ]

    rul_table.append({
        "unit": int(unit),
        "predicted_RUL": predicted_rul
    })

    prediction_details.append({

        "test_unit":
            unit,

        "window_length":
            len(hi_curve),

        "predicted_rul":
            predicted_rul,

        "test_hi":
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

    print(
        f"Unit {unit}: predicted RUL = {predicted_rul:.2f} cycles"
    )

# =====================================================
# SAVE RESULTS
# =====================================================

rul_df = pd.DataFrame(rul_table)

rul_df.to_excel(
    "test_rul_predictions.xlsx",
    index=False
)

with open(
    "test_debug_data.pkl",
    "wb"
) as f:

    pickle.dump(
        prediction_details,
        f
    )

print()
print(
    f"Predicciones generadas para {len(rul_df)} motores"
)
print(
    "Tabla guardada en: test_rul_predictions.xlsx"
)
