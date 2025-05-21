import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
from model import AdvancedAutoencoder  # or AdvancedAutoencoder
from dataset import MRISliceDataset
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
ENSEMBLE_SIZE = 3

test_dir = "PreProcessedData/test"
label_path = "PreProcessedData/test_labels.npy"
checkpoints_dir = "checkpoints"

test_loader = DataLoader(MRISliceDataset(test_dir), batch_size=BATCH_SIZE, shuffle=False)
true_labels = np.load(label_path)

# Load ensemble
models = []
# for i in range(ENSEMBLE_SIZE-1):
#     model = AdvancedAutoencoder().to(DEVICE)
#     model.load_state_dict(torch.load(os.path.join(checkpoints_dir, f"model_{i}.pth")))
#     model.eval()
#     models.append(model)
model = AdvancedAutoencoder().to(DEVICE)
model.load_state_dict(torch.load(os.path.join(checkpoints_dir, f"model_{0}.pth")))
model.eval()
models.append(model)

# computing ensemble scores 
def get_scores(models, dataloader):
    all_errors = []
    with torch.no_grad():
        for model in models:
            errors = []
            for batch in dataloader:
                batch = batch.to(DEVICE)
                out = model(batch)
                mse = ((batch - out) ** 2).mean(dim=(1, 2, 3))
                errors.extend(mse.cpu().numpy())
            all_errors.append(errors)
    return np.median(np.stack(all_errors), axis=0)

test_scores = get_scores(models, test_loader)

# compute prediction metrics 
threshold = np.percentile(test_scores, 75)
preds = (test_scores > threshold).astype(int)

auc = roc_auc_score(true_labels, test_scores)
precision = precision_score(true_labels, preds)
recall = recall_score(true_labels, preds)
f1 = f1_score(true_labels, preds)

print(f"AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# ROC curve
fpr, tpr, _ = roc_curve(true_labels, test_scores)
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.grid()
plt.legend()
# plt.show()
plt.savefig("falsePositiveRate.png", dpi=300)
plt.close()  # optional: close to avoid overlap in future plots


# precision recall curve
prec, rec, _ = precision_recall_curve(true_labels, test_scores)
plt.plot(rec, prec)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid()
# plt.show()
plt.savefig("precision_recall_curve.png", dpi=300)
plt.close()  # optional: close to avoid overlap in future plots

#consusion metrics
cm = confusion_matrix(true_labels, preds)
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix")
plt.grid(False)
# plt.show()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()  # optional: close to avoid overlap in future plots

# plotting heatmaps
def plot_heatmap(models, test_dir, scores, top_n=5, save_dir="heatmaps"):
    os.makedirs(save_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(test_dir) if f.endswith('.npy')])
    idx_sorted = np.argsort(-scores)

    for i in range(top_n):
        idx = idx_sorted[i]
        img = np.load(os.path.join(test_dir, files[idx]))
        input_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            recons = [model(input_tensor).squeeze(0).cpu().numpy() for model in models]
        recon = np.mean(recons, axis=0)

        error_map = (img - recon[0]) ** 2

        plt.figure(figsize=(12, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap="gray")
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(recon[0], cmap="gray")
        plt.title("Reconstruction")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(error_map, cmap="hot")
        plt.title(f"Error Heatmap\nScore: {scores[idx]:.4f}")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"heatmap_{i}.png"))
        plt.close()
plot_heatmap(models, test_dir, test_scores, top_n=5)
