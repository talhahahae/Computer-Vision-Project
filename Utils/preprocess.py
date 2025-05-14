import os
import glob
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Config
DATASET_DIR = "../Data"
SAVE_DIR = "../PreProcessedData"
SLICE_AXIS = 2  # axial slices
IMAGE_SHAPE = (128, 128)  # resize target, optional

os.makedirs(SAVE_DIR, exist_ok=True)

def normalize(img):
    img = img.astype(np.float32)
    return (img - np.mean(img)) / np.std(img)  # z-score normalization

def resize_slice(slice_2d, target_shape):
    from skimage.transform import resize
    return resize(slice_2d, target_shape, anti_aliasing=True)

def extract_slices(nifti_path):
    nii = nib.load(nifti_path)
    data = nii.get_fdata()
    data = normalize(data)

    slices = []
    for i in range(data.shape[SLICE_AXIS]):
        slice_2d = np.take(data, i, axis=SLICE_AXIS)
        slice_2d = resize_slice(slice_2d, IMAGE_SHAPE)
        slices.append(slice_2d)
    return slices

# Load all NIfTI files
nifti_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.nii.gz")))
all_slices = []

print("Extracting slices...")
for nifti_file in tqdm(nifti_files):
    slices = extract_slices(nifti_file)
    all_slices.extend(slices)

print(f"Total 2D slices extracted: {len(all_slices)}")

# Split into train/val/test (e.g., 70/15/15)
train, test = train_test_split(all_slices, test_size=0.3, random_state=42)
val, test = train_test_split(test, test_size=0.5, random_state=42)

splits = {'train': train, 'val': val, 'test': test}

print("Saving slices...")
for split_name, split_data in splits.items():
    split_dir = os.path.join(SAVE_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)
    for i, img in enumerate(split_data):
        np.save(os.path.join(split_dir, f"image_{i:04d}.npy"), img)

print("Preprocessing complete.")
