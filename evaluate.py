import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from utils import normalize_error

def compute_reconstruction_errors(model, dataloader, device):
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            output = model(batch)
            loss = torch.mean((output - batch) ** 2, dim=(1, 2, 3))  # per-image
            errors.extend(loss.cpu().numpy())
    return np.array(errors)

def evaluate_ensemble(models, dataloader, device, threshold=None, true_labels=None):
    all_errors = []
    for model in models:
        errors = compute_reconstruction_errors(model, dataloader, device)
        all_errors.append(normalize_error(errors))

    fused_scores = np.median(np.stack(all_errors), axis=0)
    
    if threshold is None:
        threshold = np.percentile(fused_scores, 95)

    predictions = (fused_scores > threshold).astype(int)

    if true_labels is not None:
        auc = roc_auc_score(true_labels, fused_scores)
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        return fused_scores, predictions, threshold, (auc, precision, recall, f1)

    return fused_scores, predictions, threshold
