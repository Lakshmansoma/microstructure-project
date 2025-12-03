import os
from typing import List, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class VoxelHDF5Dataset(Dataset):

    def __init__(
        self,
        base_dir: str,
        micro_folders: Optional[List[str]] = None,
        feature_key: str = "DistanceFrom",
        grid_size: int = 128,
    ):
    
        self.base_dir = base_dir
        self.grid_size = grid_size
        self.feature_key = feature_key

        # Decide which Micro* folders to use
        if micro_folders is None:
            micro_folders = [
                d for d in os.listdir(base_dir)
                if d.startswith("Micro") and os.path.isdir(os.path.join(base_dir, d))
            ]

        self.files: List[str] = []
        for mf in micro_folders:
            folder = os.path.join(base_dir, mf)
            for fn in os.listdir(folder):
                if fn.endswith(".hdf5"):
                    self.files.append(os.path.join(folder, fn))

        self.files.sort()

        if len(self.files) == 0:
            raise RuntimeError(f"No .hdf5 files found in {base_dir}")

        print(f"Found {len(self.files)} voxel files for feature '{feature_key}'.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]

        # Read the chosen feature from the HDF5 file
        with h5py.File(path, "r") as f:
            if self.feature_key not in f:
                raise KeyError(
                    f"Feature key '{self.feature_key}' not found in file {path}. "
                    f"Available keys: {list(f.keys())}"
                )
            data = f[self.feature_key][:]     # (N,) or (N, F)

        # Handle both 1D and 2D feature arrays: (N,) -> (N, 1)
        if data.ndim == 1:
            data = data[:, None]

        # For DistanceFrom in THIS dataset: (N, 3). Use ONLY first channel for now.
        if self.feature_key == "DistanceFrom" and data.shape[1] > 1:
            data = data[:, 0:1]   

        # N_voxels 
        N, F = data.shape
        expected = self.grid_size ** 3
        if N != expected:
            raise ValueError(
                f"Unexpected voxel count {N} in {path}, expected {expected} "
                f"for grid_size={self.grid_size}."
            )

       
        if self.feature_key in ("ngr", "DistanceFrom", "Taylor"):
            max_val = data.max()
            min_val = data.min()
            if max_val > min_val:
                data = (data - min_val) / (max_val - min_val)

        # -------- Reshape to 3D volume --------
        # (N, F) -> (D, H, W, F) -> (F, D, H, W)
        vol = data.reshape(
            self.grid_size,
            self.grid_size,
            self.grid_size,
            F,
        )
        vol = np.moveaxis(vol, -1, 0)  # (C, D, H, W)

        # Convert to float tensor
        vol = torch.from_numpy(vol).float()
        return vol
