from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class GrainSeqDataset(Dataset):
  

    def __init__(
        self,
        csv_paths: List[str],
        target_col: str = "EqVonMisesStress",
        ngr_col: str = "ngr",
        time_col: str = "fileID",
        min_timesteps: int = 2,
    ):
        self.target_col = target_col
        self.ngr_col = ngr_col
        self.time_col = time_col

        # 1) Load and concat
        dfs = [pd.read_csv(p) for p in csv_paths]
        df_all = pd.concat(dfs, ignore_index=True)

        # Drop useless index col if present
        if "Unnamed: 0" in df_all.columns:
            df_all = df_all.drop(columns=["Unnamed: 0"])

        # 2) Numeric-only features
        numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
        exclude = {target_col, ngr_col, time_col}
        feature_cols = [c for c in numeric_cols if c not in exclude]
        self.feature_cols = feature_cols

        # 3) Clean NaNs / infs
        df_all = df_all.replace([np.inf, -np.inf], np.nan)
        df_all = df_all.dropna(subset=self.feature_cols + [self.target_col])

        self.df_all = df_all

        # 4) Build sequences
        sequences_X: List[np.ndarray] = []
        sequences_y: List[float] = []

        grouped = df_all.groupby(self.ngr_col)

        for ngr_id, g in grouped:
            g = g.sort_values(self.time_col)
            if len(g) < min_timesteps:
                continue

            X = g[self.feature_cols].to_numpy(dtype=np.float32)  # (T_i, F)
            y = float(g[self.target_col].iloc[-1])

            sequences_X.append(X)
            sequences_y.append(y)

        if len(sequences_X) == 0:
            raise RuntimeError("No sequences built; check input CSVs and columns.")

        self.X_list = sequences_X
        self.y_list = sequences_y
        self.num_features = len(self.feature_cols)

        print(
            f"GrainSeqDataset: {len(self.X_list)} grains, "
            f"{self.num_features} features, target={self.target_col}"
        )

    def __len__(self) -> int:
        return len(self.X_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X = torch.from_numpy(self.X_list[idx])  # (T_i, F)
        y = torch.tensor([self.y_list[idx]], dtype=torch.float32)
        return X, y
