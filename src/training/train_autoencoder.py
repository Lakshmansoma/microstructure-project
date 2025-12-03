import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from src.dataset import VoxelHDF5Dataset
from src.models.cnn3d_ae import MicrostructureAE3D

BASE_VOXEL_DIR = "C:/Users/laksh/microstructure-project/data/raw/ss75fdg5dg-1/FCC_voxelwise"
FEATURE_KEY = "DistanceFrom"
GRID_SIZE = 128

BATCH_SIZE = 1          # 3D -> heavy
EPOCHS = 20             # give it more time
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    dataset = VoxelHDF5Dataset(
        base_dir=BASE_VOXEL_DIR,
        feature_key=FEATURE_KEY,
        grid_size=GRID_SIZE,
    )

    # Infer channels from one sample
    sample = dataset[0]
    in_channels = sample.shape[0]
    print("Sample volume shape:", sample.shape)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    model = MicrostructureAE3D(
        in_channels=in_channels,
        latent_dim=4096,
        base_channels=16,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.L1Loss()

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for x in loader:
            x = x.to(DEVICE)
            x_hat, _ = model(x)
            loss = criterion(x_hat, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        avg_loss = running_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss={avg_loss:.6f}")

        torch.save(
            model.state_dict(),
            out_dir / f"ae_{FEATURE_KEY}_epoch{epoch + 1}.pt",
        )


if __name__ == "__main__":
    main()
