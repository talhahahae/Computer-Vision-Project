import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class AdvancedAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # Input: (1, 128, 128)

        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),    # -> (32, 64, 64)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # -> (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # -> (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),

            nn.Flatten(),                                # -> (128*16*16 = 32768)
            nn.Linear(128 * 16 * 16, latent_dim),
            nn.ReLU(inplace=True)
        )

        # --- Decoder ---
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 16 * 16),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (128, 16, 16)),

            ResidualBlock(128),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # -> (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            ResidualBlock(64),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),   # -> (32, 64, 64)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            ResidualBlock(32),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),    # -> (1, 128, 128)
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out
