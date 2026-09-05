"""Dataset that turns a CSV of sequence pairs into model-ready tensors.

The CSV must have columns ``Seq1``, ``Seq2``, ``Label`` (0/1). Encoding is
delegated to whichever ``Encoder`` is passed in, so the dataset itself knows
nothing about specific encodings.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset

from .encoders import Encoder


class PairDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        encoder: Encoder,
        seq_length: int,
        num_samples: int | None = None,
    ):
        # Read only the columns we use; master CSVs may carry many extra feature
        # columns that would balloon memory and load time.
        self.df = pd.read_csv(csv_path, usecols=["Seq1", "Seq2", "Label"])
        if num_samples and num_samples < len(self.df):
            self.df = self.df.sample(n=num_samples, random_state=42).reset_index(
                drop=True
            )
        self.encoder = encoder
        self.seq_length = seq_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        matrix = self.encoder.encode(row["Seq1"], row["Seq2"], self.seq_length)
        return {
            "matrix": torch.tensor(matrix, dtype=torch.float32),
            "label": torch.tensor(row["Label"], dtype=torch.float32),
        }
