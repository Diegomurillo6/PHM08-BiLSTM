import joblib
import numpy as np
import pandas as pd

from scipy.optimize import curve_fit

# =====================================================
# PARAMETERS
# =====================================================

MODEL_FILE = "HI_linear_regression.pkl"

SENSOR_FILE = "normalized_dataset.csv"

MOVING_AVERAGE_WINDOW = 10

# =====================================================
# EXPONENTIAL FIT FUNCTION
# =====================================================

def exponential_model(
    x,
    a,
    b,
    c
):

    # return -a*np.exp(x-b) + c
    return a-(x**b) /c

# =====================================================
# LOAD LINEAR REGRESSION MODEL
# =====================================================

model = joblib.load(
    MODEL_FILE
)

print(
    "Loaded HI_linear_regression.pkl"
)

# =====================================================
# LOAD NORMALIZED SENSOR DATA
# =====================================================

sensor_df = pd.read_csv(
    SENSOR_FILE
)

print(
    "Sensor dataset shape:",
    sensor_df.shape
)

# =====================================================
# FEATURE COLUMNS
# =====================================================

feature_cols = [

    col

    for col in sensor_df.columns

    if col not in [
        "unit",
        "cycle"
    ]
]

print(
    "Number of features:",
    len(feature_cols)
)

# =====================================================
# PREDICT HI USING EQUATION (5)
# =====================================================

X = sensor_df[
    feature_cols
].values

sensor_df["HI_raw"] = model.predict(
    X
)

print(
    "Raw HI values generated"
)

# =====================================================
# MOVING AVERAGE SMOOTHING
# =====================================================

sensor_df["HI_smoothed"] = (

    sensor_df

    .groupby("unit")["HI_raw"]

    .transform(

        lambda x:

        x.rolling(
            MOVING_AVERAGE_WINDOW,
            min_periods=1
        ).mean()
    )
)

print(
    "Moving average smoothing completed"
)

# =====================================================
# CURVE FITTING
# =====================================================

library_records = []

curve_parameters = []

for unit_id in sorted(
    sensor_df["unit"].unique()
):

    print(
        f"Processing unit {unit_id}"
    )

    unit_df = (
        sensor_df[
            sensor_df["unit"] == unit_id
        ]

        .sort_values(
            "cycle"
        )

        .reset_index(
            drop=True
        )
    )

    cycles = (
        unit_df["cycle"]
        .values
    )

    hi_smoothed = (
        unit_df["HI_smoothed"]
        .values
    )

    try:

        initial_guess = [
            0.9,
            2,
            35000
        ]

        popt, _ = curve_fit(

            exponential_model,

            cycles,

            hi_smoothed,

            p0=initial_guess,

            maxfev=10000

        )

        fitted_hi = exponential_model(

            cycles,

            *popt

        )

        curve_parameters.append(                                                                                 
            {
                "unit": unit_id,
                "a": popt[0],
                "b": popt[1],
                "c": popt[2]
            }
        )

        print(
            "  Exponential fit successful"
        )

    except Exception as e:

        print(
            "  Fit failed:",
            e
        )

        fitted_hi = hi_smoothed

        curve_parameters.append(
            {
                "unit": unit_id,
                "a": np.nan,
                "b": np.nan,
                "c": np.nan
            }
        )

    temp_df = pd.DataFrame(
        {
            "unit": unit_id,

            "cycle": cycles,

            "HI_raw":
                unit_df["HI_raw"],

            "HI_smoothed":
                hi_smoothed,

            "HI_fitted":
                fitted_hi
        }
    )

    library_records.append(
        temp_df
    )

# =====================================================
# COMBINE ALL UNITS
# =====================================================

offline_hi_library = pd.concat(
    library_records,
    ignore_index=True
)

# =====================================================
# SAVE LIBRARY
# =====================================================

offline_hi_library.to_csv(
    "offline_HI_library.csv",
    index=False
)

print(
    "Saved offline_HI_library.csv"
)

# =====================================================
# SAVE CURVE PARAMETERS
# =====================================================

curve_parameters_df = pd.DataFrame(
    curve_parameters
)

curve_parameters_df.to_csv(
    "curve_fit_parameters.csv",
    index=False
)

print(
    "Saved curve_fit_parameters.csv"
)

# =====================================================
# SUMMARY
# =====================================================

print()

print(
    "Offline HI library shape:",
    offline_hi_library.shape
)

print(
    "Finished building HI library"
)


# import joblib
# import numpy as np
# import pandas as pd

# # =====================================================
# # PARAMETERS
# # =====================================================

# MODEL_FILE = "HI_linear_regression.pkl"
# SENSOR_FILE = "normalized_dataset.csv"

# MOVING_AVERAGE_WINDOW = 10

# # =====================================================
# # LOAD MODEL
# # =====================================================

# model = joblib.load(MODEL_FILE)

# print("Loaded HI_linear_regression.pkl")

# # =====================================================
# # LOAD SENSOR DATA
# # =====================================================

# sensor_df = pd.read_csv(SENSOR_FILE)

# print("Sensor dataset shape:", sensor_df.shape)

# # =====================================================
# # FEATURE COLUMNS
# # =====================================================

# feature_cols = [col for col in sensor_df.columns
#                 if col not in ["unit", "cycle"]]

# print("Number of features:", len(feature_cols))

# # =====================================================
# # PREDICT HI USING EQUATION (5)
# # =====================================================

# X = sensor_df[feature_cols].values

# sensor_df["HI_raw"] = model.predict(X)

# print("Raw HI values generated")

# # =====================================================
# # MOVING AVERAGE SMOOTHING
# # =====================================================

# sensor_df["HI_smoothed"] = (
#     sensor_df.groupby("unit")["HI_raw"]
#     .transform(lambda x:
#                x.rolling(MOVING_AVERAGE_WINDOW,
#                          min_periods=1).mean())
# )

# print("Moving average smoothing completed")

# # =====================================================
# # BUILD OFFLINE HI LIBRARY
# # =====================================================

# offline_hi_library = sensor_df[
#     ["unit", "cycle", "HI_raw", "HI_smoothed"]
# ].copy()

# offline_hi_library.to_csv(
#     "offline_HI_library.csv",
#     index=False
# )

# print("Saved offline_HI_library.csv")

# # =====================================================
# # SAVE INDIVIDUAL CURVES
# # =====================================================

# for unit_id in sorted(offline_hi_library["unit"].unique()):

#     unit_df = (
#         offline_hi_library[
#             offline_hi_library["unit"] == unit_id
#         ]
#         .sort_values("cycle")
#     )

#     filename = f"HI_curve_unit_{unit_id}.csv"

#     unit_df.to_csv(filename, index=False)

# print("Individual HI curves saved")

# # =====================================================
# # SUMMARY
# # =====================================================

# print()
# print("Offline HI library shape:", offline_hi_library.shape)
# print("Finished building HI library")