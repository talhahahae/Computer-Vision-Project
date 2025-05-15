import os
import numpy as np
import random
from tqdm import tqdm
from skimage.draw import disk, rectangle
from skimage.util import random_noise

# Configuration
SOURCE_DIR = "PreProcessedData/test"
OUTPUT_DIR = "PreProcessedData/synthetic_test"
LABEL_FILE = "PreProcessedData/test_labels.npy"
ANOMALY_RATIO = 0.3

os.makedirs(OUTPUT_DIR, exist_ok=True)

def add_blob_anomaly(image):
    img = image.copy()
    h, w = img.shape
    r, c = random.randint(20, h-20), random.randint(20, w-20)
    radius = random.randint(5, 15)
    rr, cc = disk((r, c), radius, shape=img.shape)
    img[rr, cc] = 1.0  # bright circle
    return img

def add_noise_anomaly(image):
    return random_noise(image, mode='s&p', amount=0.1)

def add_occlusion(image):
    img = image.copy()
    h, w = img.shape
    start = (random.randint(0, h//2), random.randint(0, w//2))
    extent = (random.randint(10, 30), random.randint(10, 30))
    rr, cc = rectangle(start=start, extent=extent, shape=img.shape)
    img[rr, cc] = 0.0  # black patch
    return img

def inject_anomaly(image):
    choice = random.choice(['blob', 'noise', 'occlusion'])
    if choice == 'blob':
        return add_blob_anomaly(image)
    elif choice == 'noise':
        return add_noise_anomaly(image)
    else:
        return add_occlusion(image)

# Load and process all test slices
all_files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith('.npy')])
labels = []

print(f"Generating synthetic anomalies from {len(all_files)} test samples...")
for f in tqdm(all_files):
    img = np.load(os.path.join(SOURCE_DIR, f))
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Decide if this will be anomalous
    if random.random() < ANOMALY_RATIO:
        img = inject_anomaly(img)
        labels.append(1)
    else:
        labels.append(0)

    np.save(os.path.join(OUTPUT_DIR, f), img)

# Save labels
np.save(LABEL_FILE, np.array(labels))
print(f"✅ Synthetic test set saved to: {OUTPUT_DIR}")
print(f"✅ Labels saved to: {LABEL_FILE}")
print(f"Total anomalies: {sum(labels)} / {len(labels)}")
