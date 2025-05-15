import torch
from torch.utils.data import DataLoader
from dataset import MRISliceDataset
from model import AdvancedAutoencoder
from train import train_model
from evaluate import evaluate_ensemble
from tensorboard_logger import get_tensorboard_writer
import os
import numpy as np
from utils import seed_everything

# Configs
BATCH_SIZE = 32
EPOCHS = 20
ENSEMBLE_SIZE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    seed_everything(42)

    # Data paths
    train_dir = "PreProcessedData/train"
    val_dir = "PreProcessedData/val"
    test_dir = "PreProcessedData/synthetic_test"
    labels_file = "PreProcessedData/test_labels.npy"  # optional

    # Load data
    train_loader = DataLoader(MRISliceDataset(train_dir), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MRISliceDataset(val_dir), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(MRISliceDataset(test_dir), batch_size=BATCH_SIZE, shuffle=False)

    # TensorBoard
    writer = get_tensorboard_writer(tag="mri_ensemble")

    # Train ensemble
    models = []
    for i in range(ENSEMBLE_SIZE):
        model = AdvancedAutoencoder().to(DEVICE)
        model = train_model(model, train_loader, DEVICE, epochs=EPOCHS, model_idx=i, log_writer=writer)
        models.append(model)

    # Evaluate
    labels = np.load(labels_file) if os.path.exists(labels_file) else None
    scores, preds, threshold, metrics = evaluate_ensemble(models, test_loader, DEVICE, true_labels=labels)

    # Log to tensorboard
    if metrics:
        auc, precision, recall, f1 = metrics
        writer.add_scalar("Eval/AUC", auc)
        writer.add_scalar("Eval/Precision", precision)
        writer.add_scalar("Eval/Recall", recall)
        writer.add_scalar("Eval/F1", f1)

        print(f"Threshold: {threshold:.4f}")
        print(f"AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    else:
        print(f"Threshold: {threshold:.4f} | No labels provided for metrics.")

    writer.close()

if __name__ == "__main__":
    main()
