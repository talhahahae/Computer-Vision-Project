import numpy as np
import matplotlib.pyplot as plt

# Load a .npy file
img = np.load("../PreProcessedData/train/image_0193.npy")

# Normalize for viewing
def normalize(img):
    img = img - np.min(img)
    img = img / np.max(img)
    return img

img_norm = normalize(img)

# Save as PNG
plt.imsave("output_image.png", img_norm, cmap='gray')
print("Saved to output_image.png")