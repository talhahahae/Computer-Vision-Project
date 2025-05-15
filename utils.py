import numpy as np
import torch
import random

def compute_threshold(errors, percentile=95):
    return np.percentile(errors, percentile)

def normalize_error(errors):
    return (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)

def get_reconstruction_errors(model, dataloader, device):
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            out = model(batch)
            loss = ((batch - out) ** 2).mean(dim=[1, 2, 3])
            errors.extend(loss.cpu().numpy())
    return np.array(errors)

def seed_everything(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False