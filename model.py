import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class AdvancedAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        # Input assumed: (1, 128, 128)

        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),   # (16, 64, 64)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            ResidualBlock(16),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32),

            nn.Flatten(),                                           # (32*32*32)
            nn.Linear(32 * 32 * 32, latent_dim),                    # (latent_dim)
            nn.ReLU()
        )

        # --- Decoder ---
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 32 * 32),                    # (32*32*32)
            nn.ReLU(),
            nn.Unflatten(1, (32, 32, 32)),                          # (32, 32, 32)

            ResidualBlock(32),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # (16, 64, 64)
            nn.BatchNorm2d(16),
            nn.ReLU(),

            ResidualBlock(16),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # (1, 128, 128)
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
