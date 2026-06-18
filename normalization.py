import joblib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==========================
# Load C-MAPSS data
# ==========================

df = pd.read_csv(
    "train70.txt",
    sep=r"\s+",
    header=None
)

# print('First rows of df: ',df)

# ==========================
# Column definitions
# ==========================

unit_col = 0
df.rename(columns={df.columns[0]: "unit"}, inplace=True)

cycle_col = 1
df.rename(columns={df.columns[1]: "cycle"}, inplace=True)

operating_cols = [2, 3, 4]

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
 


# ==========================
# K-means operating conditions
# ==========================

kmeans = KMeans(
    n_clusters=6,
    random_state=0,
    n_init=20
)


df["cluster"] = kmeans.fit_predict(
    df.iloc[:, operating_cols]
)

# =====================================================
# SAVE TRAINED KMEANS MODEL
# =====================================================

joblib.dump(
    kmeans,
    "operating_condition_kmeans.pkl"
)

print(
    "Saved operating_condition_kmeans.pkl"
)

# print('cluster centers: ', kmeans.cluster_centers_)
# print('Clusters: ', df["cluster"])


# ==========================
# Create normalized copy
# ==========================

df_norm = df.copy()

df_norm[sensor_cols] = df_norm[sensor_cols].astype(float)

# Normalize selected sensors
# independently inside each cluster

normalization_records = []

for cluster_id in df["cluster"].unique():

    mask = df["cluster"] == cluster_id

    cluster_data = df.loc[
        mask,
        sensor_cols
    ]

    means = cluster_data.mean()
    stds = cluster_data.std()

    stds = stds.replace(0,1)

    # -----------------------------------------
    # Save normalization statistics
    # -----------------------------------------

    for sensor in sensor_cols:

        normalization_records.append(
            {
                "cluster": int(cluster_id),
                "sensor": int(sensor),
                "mean": means[sensor],
                "std": stds[sensor]
            }
        )

    # -----------------------------------------
    # Normalize sensors
    # -----------------------------------------

    df_norm.loc[
        mask,
        sensor_cols
    ] = (
        cluster_data - means
    ) / stds

# =====================================================
# SAVE NORMALIZATION PARAMETERS
# =====================================================

norm_df = pd.DataFrame(
    normalization_records
)

norm_df.to_csv(
    "cluster_normalization.csv",
    index=False
)

print(
    "Saved cluster_normalization.csv"
)


# ==========================
# Keep only useful columns
# ==========================

final_df = pd.concat(
    [
        df_norm.iloc[:, [unit_col, cycle_col]],
        df_norm.iloc[:, sensor_cols]
    ],
    axis=1
)

print('Final df debugging:')
print(final_df[sensor_cols].mean().mean())
print(final_df[sensor_cols].std().mean())

print(final_df.isna().sum().sum())
print(final_df.columns)


# Save for later windowing
final_df.to_csv(
    "normalized_dataset.csv",
    index=False
)

# Visualizar los 6 clusters de condiciones de operacion

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    df.iloc[:,2],
    df.iloc[:,3],
    df.iloc[:,4],
    c=df['cluster'],
    s=2
)

ax.set_xlabel('Setting 1')
ax.set_ylabel('Setting 2')
ax.set_zlabel('Setting 3')

plt.show()

# # Crear ventanas para entrenamiento de autoencoder ===============================
# windows = []
# WINDOW_SIZE = 5

# for unit_id, group in final_df.groupby("unit"):

#     group = group.sort_values("cycle")

#     X = group[sensor_cols].values

#     for i in range(len(X) - WINDOW_SIZE + 1):

#         window = X[i:i + WINDOW_SIZE]

#         windows.append(window)

# windows = np.array(windows, dtype=np.float32)

# print("Window tensor shape:")
# print(windows.shape)

# # Expected:
# # (num_windows, WINDOW_SIZE, 14)



# print(windows.min())
# print(windows.max())
# print(windows.mean())
# print(windows.std())

# =====================================================
# CREATE WINDOWS + METADATA
# =====================================================

windows = []
window_metadata = []

WINDOW_SIZE = 5

for unit_id, group in final_df.groupby("unit"):

    group = group.sort_values("cycle")

    X = group[sensor_cols].values
    cycles = group["cycle"].values

    for i in range(len(X) - WINDOW_SIZE + 1):

        window = X[i:i + WINDOW_SIZE]

        windows.append(window)

        window_metadata.append(
            {
                "unit": int(unit_id),
                "window_idx": int(i),
                "cycle_end": int(
                    cycles[i + WINDOW_SIZE - 1]
                )
            }
        )

windows = np.array(
    windows,
    dtype=np.float32
)

metadata_df = pd.DataFrame(
    window_metadata
)

print("Window tensor shape:")
print(windows.shape)

# -----------------------------------------------------
# SAVE WINDOWS AND METADATA
# -----------------------------------------------------

np.save(
    "training_windows.npy",
    windows
)

metadata_df.to_csv(
    "window_metadata.csv",
    index=False
)

print(
    f"Saved {len(windows)} windows"
)

# Entrenamiento del autoencoder

# =====================================================
# PARAMETERS
# =====================================================

HIDDEN_SIZE = 200
BATCH_SIZE = 40
LEARNING_RATE = 0.02
EPOCHS = 6

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =====================================================
# DATASET
# =====================================================

class WindowDataset(Dataset):

    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):

        x = self.windows[idx]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(x, dtype=torch.float32)
        )


dataset = WindowDataset(windows)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# =====================================================
# BILSTM AUTOENCODER
# =====================================================

class BiLSTMAutoencoder(nn.Module):

    def __init__(
        self,
        input_size=14,
        hidden_size=200,
        window_size=WINDOW_SIZE
    ):
        super().__init__()

        self.window_size = window_size
        self.hidden_size = hidden_size

        # ---------------------
        # Encoder
        # ---------------------

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # ---------------------
        # Decoder
        # ---------------------

        self.decoder = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # ---------------------
        # Output Layer
        # ---------------------

        self.output_layer = nn.Linear(
            hidden_size * 2,
            input_size
        )

    def forward(self, x):

        # =====================================
        # ENCODER
        # =====================================

        _, (h_n, _) = self.encoder(x)

        # h_n shape:
        # (2, batch_size, hidden_size)

        forward_hidden = h_n[0]
        backward_hidden = h_n[1]

        embedding = torch.cat(
            [forward_hidden, backward_hidden],
            dim=1
        )

        # embedding shape:
        # (batch_size, 400)

        # =====================================
        # REPEAT EMBEDDING
        # =====================================

        decoder_input = embedding.unsqueeze(1)

        decoder_input = decoder_input.repeat(
            1,
            self.window_size,
            1
        )

        # shape:
        # (batch_size, window_size, 400)

        # =====================================
        # DECODER
        # =====================================

        decoded, _ = self.decoder(
            decoder_input
        )

        # shape:
        # (batch_size, window_size, 400)

        # =====================================
        # RECONSTRUCTION
        # =====================================

        output = self.output_layer(decoded)

        # shape:
        # (batch_size, window_size, 14)

        return output


# =====================================================
# MODEL
# =====================================================

NUM_FEATURES = len(sensor_cols)

model = BiLSTMAutoencoder(
    input_size=NUM_FEATURES,
    hidden_size=HIDDEN_SIZE,
    window_size=WINDOW_SIZE
)

model = model.to(DEVICE)

print(model)


# =====================================================
# LOSS FUNCTION
# =====================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =====================================================
# TRAINING
# =====================================================

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for batch_x, batch_y in loader:

        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        reconstruction = model(batch_x)

        loss = criterion(
            reconstruction,
            batch_y
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS}"
        f" - Loss: {avg_loss:.6f}"
    )


# =====================================================
# SAVE MODEL
# =====================================================

torch.save(
    model.state_dict(),
    f"bilstm_autoencoder_loss_{avg_loss:.6f}.pt"
)

print("Training complete.")