import torch
import torch.nn as nn


class MicrostructureAE3D(nn.Module):
   
    def __init__(self, in_channels: int = 1, latent_dim: int = 4096, base_channels: int = 16):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8   # last encoder channels

        # 128 -> 64
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, c1, 3, stride=2, padding=1),
            nn.BatchNorm3d(c1),
            nn.ReLU(inplace=True),
        )
        # 64 -> 32
        self.enc2 = nn.Sequential(
            nn.Conv3d(c1, c2, 3, stride=2, padding=1),
            nn.BatchNorm3d(c2),
            nn.ReLU(inplace=True),
        )
        # 32 -> 16
        self.enc3 = nn.Sequential(
            nn.Conv3d(c2, c3, 3, stride=2, padding=1),
            nn.BatchNorm3d(c3),
            nn.ReLU(inplace=True),
        )
        # 16 -> 8
        self.enc4 = nn.Sequential(
            nn.Conv3d(c3, c4, 3, stride=2, padding=1),
            nn.BatchNorm3d(c4),
            nn.ReLU(inplace=True),
        )

        self.last_channels = c4
        self.enc_out_dim = self.last_channels * 8 * 8 * 8  # after 4 downsamples from 128

        self.fc_enc = nn.Linear(self.enc_out_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.enc_out_dim)

        # Decoder: 8 -> 16 -> 32 -> 64 -> 128
        self.dec1 = nn.Sequential(
            nn.ConvTranspose3d(self.last_channels, c3, 4, stride=2, padding=1),
            nn.BatchNorm3d(c3),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(c3, c2, 4, stride=2, padding=1),
            nn.BatchNorm3d(c2),
            nn.ReLU(inplace=True),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose3d(c2, c1, 4, stride=2, padding=1),
            nn.BatchNorm3d(c1),
            nn.ReLU(inplace=True),
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose3d(c1, in_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = x.view(x.size(0), -1)
        z = self.fc_enc(x)
        return z

    def decode(self, z):
        x = self.fc_dec(z)
        x = x.view(-1, self.last_channels, 8, 8, 8)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
