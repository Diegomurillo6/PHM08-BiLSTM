# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # =====================================================
# # LOAD HI CURVES
# # =====================================================

# hi_df = pd.read_csv(
#     "hi_curves.csv"
# )

# print(hi_df.head())

# # =====================================================
# # PLOT ALL HI CURVES
# # =====================================================

# plt.figure(figsize=(12,6))

# for unit_id, group in hi_df.groupby("unit"):

#     group = group.sort_values(
#         "window_idx"
#     )

#     plt.plot(
#         group["window_idx"],
#         group["HI_smoothed"],
#         alpha=0.4
#     )

# plt.xlabel("Window Index")
# plt.ylabel("Health Index")
# plt.title("All HI Curves")
# plt.grid(True)

# plt.show()

# # Plotting random curves =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

hi_df = pd.read_csv(
    "hi_curves.csv"
)
hi_df = pd.read_csv(
    "offline_HI_library.csv"
)
# hi_df = pd.read_csv(
#     "cycle_level_hi.csv"
# )

# =====================================================
# RANDOM SAMPLE OF ENGINES
# =====================================================

N_UNITS = 5

units = np.random.choice(
    hi_df["unit"].unique(),
    size=min(
        N_UNITS,
        hi_df["unit"].nunique()
    ),
    replace=False
)

available_units = hi_df["unit"].unique()

START_UNIT = 1
END_UNIT  = 5
units = [
        u
        for u in range(
            START_UNIT,
            END_UNIT + 1
        )
        if u in available_units
    ]

plt.figure(figsize=(6,6))

for unit_id in units:

    group = (
        hi_df[
            hi_df["unit"] == unit_id
        ]
        .sort_values(
            # "window_idx"
            "cycle" # for cycle based HI curves from LR model
        )
    )

    plt.plot(
        # group["window_idx"],
        group["cycle"], # for cycle based HI curves from LR model
        group["HI_smoothed"],
        # group["HI_fitted"], # for cycle based HI curves from LR model
        # group["HI_cycle"], # for cycle based HI curves from LR model
        label=f"Unit {unit_id}"
    )

    plt.plot(
        # group["window_idx"],
        group["cycle"], # for cycle based HI curves from LR model
        # group["HI_smoothed"],
        group["HI_fitted"], # for cycle based HI curves from LR model
        # group["HI_cycle"], # for cycle based HI curves from LR model
        label=f"Unit {unit_id}"
    )

plt.xlabel("Window Index")
# plt.xlabel("Cycle") # for cycle based HI curves from LR model
plt.ylabel("Health Index")
plt.title("Sample HI Curves")
# plt.legend()
plt.grid(True)

plt.show()

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# hi_df = pd.read_csv(
#     "hi_curves.csv"
# )


# # =====================================================
# # UNIT SELECTION
# # =====================================================

# USE_RANDOM = True
# USE_RANGE = False
# USE_CUSTOM_LIST = False

# # ----- Random mode -----
# N_UNITS = 30

# # ----- Range mode -----
# START_UNIT = 6
# END_UNIT = 15

# # ----- Custom list mode -----
# UNITS_TO_PLOT = [
#     1,
#     5,
#     12,
#     37,
#     68
# ]

# # =====================================================
# # BUILD UNIT LIST
# # =====================================================

# available_units = hi_df["unit"].unique()

# if USE_RANDOM:

#     units = np.random.choice(
#         available_units,
#         size=min(
#             N_UNITS,
#             len(available_units)
#         ),
#         replace=False
#     )

# elif USE_RANGE:

#     units = [
#         u
#         for u in range(
#             START_UNIT,
#             END_UNIT + 1
#         )
#         if u in available_units
#     ]

# elif USE_CUSTOM_LIST:

#     units = [
#         u
#         for u in UNITS_TO_PLOT
#         if u in available_units
#     ]

# else:

#     raise ValueError(
#         "Select one unit selection mode."
#     )

# plt.figure(figsize=(12,6))

# for unit_id in units:

#     group = (
#         hi_df[
#             hi_df["unit"] == unit_id
#         ]
#         .sort_values(
#             "window_idx"
#         )
#     )

#     plt.plot(
#         group["window_idx"],
#         group["HI_smoothed"],
#         label=f"Unit {unit_id}"
#     )

# plt.xlabel("Window Index")
# plt.ylabel("Health Index")
# plt.title("HI Curves")
# plt.legend()
# plt.grid(True)

# plt.show()
