import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MRISliceDataset(Dataset):
    def __init__(self, directory):
        self.file_paths = sorted([
            os.path.join(directory, fname)
            for fname in os.listdir(directory)
            if fname.endswith(".npy")
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = np.load(self.file_paths[idx])
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # Normalize to [0, 1]
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Shape: (1, H, W)
        return img
