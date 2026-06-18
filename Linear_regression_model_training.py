import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score
)

# =====================================================
# PARAMETERS
# =====================================================

WINDOW_SIZE = 5

HI_FILE = "hi_curves.csv" # Tiene datos de HI por ventana, con y sin media movil

SENSOR_FILE = "normalized_dataset.csv" # Contiene el dataset original normalizado

# =====================================================
# LOAD DATA
# =====================================================

hi_df = pd.read_csv(
    HI_FILE
)

sensor_df = pd.read_csv(
    SENSOR_FILE
)

# =====================================================
# WINDOW HI -> CYCLE HI
# =====================================================

# Convierte HI de las ventanas a HI de ciclo
# Toma promedio de todas las ventanas que usaron el ciclo dado en el embedding

cycle_records = []

for unit_id in sorted(
    hi_df["unit"].unique()
):

    unit_windows = (
        hi_df[
            hi_df["unit"] == unit_id
        ]
        .sort_values(
            "window_idx"
        )
    )

    max_cycle = (
        unit_windows["cycle_end"]
        .max()
    )

    cycle_hi_lists = {

        cycle: []

        for cycle in range(
            1,
            max_cycle + 1
        )
    }

    for _, row in unit_windows.iterrows():

        cycle_end = int(
            row["cycle_end"]
        )

        hi_value = float(
            row["HI_smoothed"]
        )

        cycle_start = (
            cycle_end
            - WINDOW_SIZE
            + 1
        )

        for cycle in range(
            cycle_start,
            cycle_end + 1
        ):

            cycle_hi_lists[
                cycle
            ].append(
                hi_value
            )

    for cycle, values in (
        cycle_hi_lists.items()
    ):

        cycle_records.append(
            {
                "unit": unit_id,
                "cycle": cycle,
                "HI_cycle": np.mean(
                    values
                )
            }
        )

cycle_hi_df = pd.DataFrame(
    cycle_records
)

# Guarda el dataframe de HI por ciclo

cycle_hi_df.to_csv(
    "cycle_level_hi.csv",
    index=False
)

print(
    "Cycle-level HI saved"
)

# =====================================================
# MERGE SENSOR DATA + HI
# =====================================================

lr_dataset = pd.merge(
    sensor_df,
    cycle_hi_df,
    on=[
        "unit",
        "cycle"
    ],
    how="inner"
)

lr_dataset.to_csv(
    "lr_training_dataset.csv",
    index=False
)

print(
    "Merged dataset shape:",
    lr_dataset.shape
)

# =====================================================
# FEATURES
# =====================================================

feature_cols = [

    col

    for col in lr_dataset.columns

    if col not in [
        "unit",
        "cycle",
        "HI_cycle"
    ]
]

# Extrae informaicón para entrenar modelo de regresion linear, sensores para x y HI para y

X = lr_dataset[feature_cols].values

y = lr_dataset["HI_cycle"].values

print(
    "Number of features:",
    len(feature_cols)
)

# =====================================================
# TRAIN LINEAR REGRESSION
# =====================================================

model = LinearRegression(tol=1e-7)

model.fit(X,y)

# =====================================================
# SAVE TRAINED LR MODEL
# =====================================================

joblib.dump(
    model,
    "HI_linear_regression.pkl"
)

print(
    "Saved HI_linear_regression.pkl"
)

# =====================================================
# SAVE EQUATION (5)
# =====================================================

# Ecuación 5 es la que hace el mapeo de un ciclo cualquiera a un valor de HI
# sin necesitar usar los modelos directamente, solo los resultados de parametros 
# para mapear entrada a HI

# Guardar para cada sensor, el coeficiente que mapea su valor a un valor de HI
# que se suma con el resto de sensores.
coef_df = pd.DataFrame(
    {
        "feature": feature_cols,
        "beta": model.coef_
    }
)

coef_df.to_csv(
    "equation5_coefficients.csv",
    index=False
)

pd.DataFrame(
    {
        "beta0": [
            model.intercept_
        ]
    }
).to_csv(
    "equation5_intercept.csv",
    index=False
)

print(
    "Saved Equation 5 coefficients"
)

# =====================================================
# VISUALIZE COEFFICIENTS OF EQUATION (5)
# =====================================================

print("\nIntercept (β0):")
print(model.intercept_)

print("\nCoefficients:")

for feature, coef in zip(feature_cols, model.coef_):
    print(f"{feature}: {coef:.8f}")


# =====================================================
# EVALUATE THE RESULTING MAPPING USING THE GIVEN 
# =====================================================

H_LR = model.predict(X)

# =====================================================
# TARGET H_TG
# =====================================================

H_TG = y

# =====================================================
# COST FUNCTION
# Eq.:
# Σ(H_LR - H_TG)^2
# =====================================================

SSE = np.sum((H_LR - H_TG) ** 2)

print("\nEquation (5) Cost:")

print(f"SSE = {SSE:.6f}")

# =====================================================
# OTHER METRICS
# =====================================================

MSE = mean_squared_error(
    H_TG,
    H_LR
)

RMSE = np.sqrt(
    MSE
)

R2 = r2_score(
    H_TG,
    H_LR
)

print(
    f"MSE  = {MSE:.6f}"
)

print(
    f"RMSE = {RMSE:.6f}"
)

print(
    f"R²   = {R2:.6f}"
)

# =====================================================
# SAVE PREDICTIONS
# =====================================================

results_df = lr_dataset[
    ["unit", "cycle"]
].copy()

results_df["HI_target"] = H_TG

results_df["HI_predicted"] = H_LR

results_df["error"] = (
    H_LR - H_TG
)

results_df.to_csv(
    "lr_predictions.csv",
    index=False
)

print(
    "\nPredictions saved"
)

