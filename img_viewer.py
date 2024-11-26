import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Define file paths
image_path = 'data/informal_classification/India/Mumbai/S2/training/Mumbai.tif'
mask_path = 'data/informal_classification/India/Mumbai/S2/training/Mumbai_ground_truth.tif'

# Load and display the input image
with rasterio.open(image_path) as src:
    mumbai_image = src.read()  # Shape: (Bands, Height, Width)

# If the image has multiple bands, show the first three as RGB if available
if mumbai_image.shape[0] >= 3:  # Ensure there are at least 3 bands for RGB
    rgb_image = np.stack([
        mumbai_image[0],  # Red channel
        mumbai_image[1],  # Green channel
        mumbai_image[2]   # Blue channel
    ], axis=-1)
else:
    # Use the first band if there are fewer than 3 bands
    rgb_image = mumbai_image[0]

plt.figure(figsize=(10, 10))
plt.title("Mumbai Input Image")
plt.imshow(rgb_image / rgb_image.max())  # Normalize to [0, 1] for display
plt.axis('off')
plt.show()

# Load and display the ground truth mask
with rasterio.open(mask_path) as src:
    mumbai_mask = src.read(1)  # Read the first band

plt.figure(figsize=(10, 10))
plt.title("Mumbai Ground Truth Mask")
plt.imshow(mumbai_mask, cmap='tab20')  # Use a colormap for discrete classes
plt.axis('off')
plt.show()
