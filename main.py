import torch
from torch.utils.data import DataLoader
from dataset import MRISliceDataset
from model import AdvancedAutoencoder
from train_resume import train_model
from evaluate import evaluate_ensemble
from tensorboard_logger import get_tensorboard_writer
import os
import numpy as np
from utils import seed_everything, compute_threshold
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Configs
BATCH_SIZE = 32
EPOCHS = 5
ENSEMBLE_SIZE = 3
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
        model = AdvancedAutoencoder(latent_dim=128).to(DEVICE)
        model = train_model(model, train_loader, DEVICE, epochs=EPOCHS, model_idx=i, log_writer=writer)

        #  Save model checkpoint
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_{i}.pth"))

        models.append(model)

    # Evaluate
    labels = np.load(labels_file) if os.path.exists(labels_file) else None
    # Evaluate reconstruction performance on validation set
    val_scores, _, _ = evaluate_ensemble(models, val_loader, DEVICE, true_labels=None)

    # Log reconstruction statistics
    mean_val = np.mean(val_scores)
    std_val = np.std(val_scores)
    threshold = compute_threshold(val_scores, percentile=95)

    print(f"\n Validation Reconstruction Stats:")
    print(f"Mean Error: {mean_val:.4f}")
    print(f"Std. Dev  : {std_val:.4f}")
    print(f"95th Percentile Threshold: {threshold:.4f}")
    print('validation score', val_scores);
    threshold = compute_threshold(val_scores, percentile=95)
    print(f" Selected threshold from validation set: {threshold:.4f}")

    # --- Step 2: Run on test set ---
    print("\n Running inference on test set...")
    labels = np.load(labels_file) if os.path.exists(labels_file) else None
    test_scores, test_preds, _, metrics = evaluate_ensemble(models, test_loader, DEVICE, true_labels=labels, threshold=threshold)
    
    # Log to tensorboard
    if metrics:
        auc, precision, recall, f1 = metrics
        writer.add_scalar("Eval/AUC", auc)
        writer.add_scalar("Eval/Precision", precision)
        writer.add_scalar("Eval/Recall", recall)
        writer.add_scalar("Eval/F1", f1)

        prec, rec, _ = precision_recall_curve(labels, test_scores)
        plt.figure()
        plt.plot(rec, prec, label=f'F1: {f1:.2f}')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.grid(True)
        plt.legend()
        plt.savefig("pr_curve_train.png", dpi=300)
        plt.close()

        fpr, tpr, _ = roc_curve(labels, test_scores)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC: {auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--', label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.grid(True)
        plt.legend()
        plt.savefig("roc_curve_train.png", dpi=300)
        plt.close()

        cm = confusion_matrix(labels, test_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix_train.png", dpi=300)
        plt.close()

        print(f"Threshold: {threshold:.4f}")
        print(f"AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    else:
        print(f"Threshold: {threshold:.4f} | No labels provided for metrics.")
        print("\n No labels available — showing unsupervised stats only.")
        print(f"Mean score: {np.mean(test_scores):.4f}")
        print(f"Max score: {np.max(test_scores):.4f}")
        print(f"Samples predicted as anomaly: {(test_preds == 1).sum()} / {len(test_preds)}")


    writer.close()

if __name__ == "__main__":
    main()
