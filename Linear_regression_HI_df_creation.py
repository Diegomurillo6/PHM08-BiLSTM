import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =====================================================
# PARAMETERS
# =====================================================

MODEL_PATH = "bilstm_autoencoder_loss_0.013278.pt"

WINDOWS_FILE = "training_windows.npy"
METADATA_FILE = "window_metadata.csv"

WINDOW_SIZE = 5
NUM_FEATURES = 14
HIDDEN_SIZE = 200

BATCH_SIZE = 128

N_HEALTHY_WINDOWS = 3

SMOOTHING_WINDOW = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
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

        return torch.tensor(
            self.windows[idx],
            dtype=torch.float32
        )


# =====================================================
# MODEL
# =====================================================

class BiLSTMAutoencoder(nn.Module):

    def __init__(
        self,
        input_size=14,
        hidden_size=200,
        window_size=5
    ):

        super().__init__()

        self.window_size = window_size

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

        self.decoder = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

        self.output_layer = nn.Linear(
            hidden_size * 2,
            input_size
        )

    def encode(self, x):

        _, (h_n, _) = self.encoder(x)

        forward_hidden = h_n[0]
        backward_hidden = h_n[1]

        embedding = torch.cat(
            [
                forward_hidden,
                backward_hidden
            ],
            dim=1
        )

        return embedding

    def forward(self, x):

        embedding = self.encode(x)

        decoder_input = (
            embedding
            .unsqueeze(1)
            .repeat(
                1,
                self.window_size,
                1
            )
        )

        decoded, _ = self.decoder(
            decoder_input
        )

        output = self.output_layer(
            decoded
        )

        return output


# =====================================================
# LOAD WINDOW DATA
# =====================================================

windows = np.load(
    WINDOWS_FILE
)

# El metadata tiene info de cual ID etc corresponde a cual ventana
metadata_df = pd.read_csv(
    METADATA_FILE
)

print(
    "Loaded windows:",
    windows.shape
)

dataset = WindowDataset(
    windows
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =====================================================
# LOAD MODEL
# =====================================================

model = BiLSTMAutoencoder(
    input_size=NUM_FEATURES,
    hidden_size=HIDDEN_SIZE,
    window_size=WINDOW_SIZE
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)

model.eval()

print("Model loaded")


# =====================================================
# CREATE EMBEDDINGS
# =====================================================

all_embeddings = []

with torch.no_grad():

    for batch_x in loader:

        batch_x = batch_x.to(
            DEVICE
        )

        embeddings = model.encode(
            batch_x
        )

        all_embeddings.append(
            embeddings
            .cpu()
            .numpy()
        )

all_embeddings = np.vstack(
    all_embeddings
)

print(
    "Embedding matrix:",
    all_embeddings.shape
)


# =====================================================
# SAVE EMBEDDINGS
# =====================================================

embedding_cols = [
    f"emb_{i}"
    for i in range(
        all_embeddings.shape[1]
    )
]

embeddings_df = pd.DataFrame(
    all_embeddings,
    columns=embedding_cols
)

embeddings_df = pd.concat(
    [
        metadata_df,
        embeddings_df
    ],
    axis=1
)

embeddings_df.to_csv(
    "embeddings.csv",
    index=False
)

print(
    "Embeddings saved"
)


# =====================================================
# BUILD Znorm
# =====================================================

# ES el conjunto de ventanas al inicio de cada prueba que se consideran en estado saludable

normal_embeddings = []

for unit_id in sorted(
    metadata_df["unit"].unique()
):

    unit_rows = (
        embeddings_df[
            embeddings_df["unit"]
            == unit_id
        ]
        .sort_values(
            "window_idx"
        )
    )

    z = unit_rows[
        embedding_cols
    ].values

    normal_embeddings.append(
        z[:N_HEALTHY_WINDOWS]
    )

Znorm = np.vstack(
    normal_embeddings
)

np.save(
    "normal_embeddings.npy",
    Znorm
)

print(
    "Znorm shape:",
    Znorm.shape
)


# =====================================================
# EQUATION 6
# =====================================================

# Ecuacion 6 calcula la desviación de una ventana con respecto al HI del estado saludable

dist_matrix = cdist(
    all_embeddings,
    Znorm,
    metric="euclidean"
)

deviations = (
    dist_matrix.mean(
        axis=1
    )
)

print(
    "Deviation vector:",
    deviations.shape
)


# =====================================================
# EQUATION 7
# =====================================================

# Usa el resultado de las desviaciones para normalizar el HI y la desviacion a un rango de 1 a 0

HI = np.zeros(
    len(deviations)
)

for unit_id in sorted(
    metadata_df["unit"].unique()
):

    mask = (
        metadata_df["unit"]
        == unit_id
    )

    d = deviations[mask]

    d_min = d.min()
    d_max = d.max()

    if d_max == d_min:

        HI[mask] = 1.0

    else:

        HI[mask] = (
            d_max - d
        ) / (
            d_max - d_min
        )


# =====================================================
# SAVE HI CURVES
# =====================================================

# Crea columna en dataframe con el valor directo del HI, y otra columna con el valor 
# usando media movil ocn ventana de 10

hi_df = metadata_df.copy()

hi_df["deviation"] = deviations

hi_df["HI"] = HI

hi_df["HI_smoothed"] = (
    hi_df
    .groupby("unit")["HI"]
    .transform(
        lambda x:
        x.rolling(
            window=SMOOTHING_WINDOW,
            min_periods=1
        ).mean()
    )
)

hi_df.to_csv(
    "hi_curves.csv",
    index=False
)

print(
    "HI curves saved"
)

print(
    hi_df.head()
)
