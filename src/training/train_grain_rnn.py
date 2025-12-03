import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from src.grain_rnn_dataset import GrainSeqDataset
from src.models.grain_lstm import GrainLSTM


def collate_fn(batch):
    # batch: list of (X, y) where X: (T_i, F)
    X_list = [b[0] for b in batch]
    y_list = [b[1] for b in batch]

    lengths = torch.tensor([x.shape[0] for x in X_list], dtype=torch.long)
    X_padded = pad_sequence(X_list, batch_first=True)  # (B, T_max, F)
    y_batch = torch.stack(y_list, dim=0)               # (B, 1)

    return X_padded, y_batch, lengths


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data" / "raw" / "ss75fdg5dg-1" / "FCC_grainwise" / "FCC_grainwise"

    csv_paths = [
        str(data_root / "micro1_all_grainwise.csv"),
        str(data_root / "micro2_all_grainwise.csv"),
        str(data_root / "micro3_all_grainwise.csv"),
        str(data_root / "micro4_all_grainwise.csv"),
        str(data_root / "micro5_all_grainwise.csv"),
        str(data_root / "micro6_all_grainwise.csv"),
    ]

    # --- dataset ---
    dataset = GrainSeqDataset(csv_paths=csv_paths, min_timesteps=2)
    input_dim = dataset[0][0].shape[1]
    print("Input dim:", input_dim)

    # --- split ---
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train

    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))

    # --- model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = GrainLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2, bidirectional=True).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 20

    for epoch in range(1, EPOCHS + 1):
        # train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch, lengths in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            y_hat = model(X_batch, lengths)
            loss = criterion(y_hat, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_ds)

        # val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch, lengths in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                lengths = lengths.to(device)

                y_hat = model(X_batch, lengths)
                loss = criterion(y_hat, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_ds)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} - "
            f"Train MSE: {train_loss:.4f} - Val MSE: {val_loss:.4f}"
        )

    # --- save model ---
    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "grain_lstm_eqvm.pt"
    torch.save(model.state_dict(), str(out_path))
    print("Saved model to:", out_path)


if __name__ == "__main__":
    main()
