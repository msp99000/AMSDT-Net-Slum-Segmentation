import rasterio
from torch.utils.data import Dataset
import os
import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations as A

class SlumSegmentationDataset(Dataset):
    def __init__(self, image_path, mask_path, transform=None, input_channels=10):
        self.image_path = image_path  # Single .tif file path for the image
        self.mask_path = mask_path    # Single .tif file path for the mask
        self.transform = transform
        self.input_channels = input_channels  # Set input_channels here

        # Open image and mask once
        with rasterio.open(self.image_path) as src:
            self.image = src.read()  # Shape: [channels, height, width]
            self.image_meta = src.meta

            # Select the first `input_channels` channels (either 3 or 10)
            self.image = self.image[:self.input_channels, :, :]

        with rasterio.open(self.mask_path) as src:
            self.mask = src.read(1)  # Shape: [height, width] (assuming mask is single-channel)
            self.mask_meta = src.meta

    def __len__(self):
        # Return 1 for the dataset since we’re no longer patching the image
        return 1

    def __getitem__(self, idx):
        # Return the entire image and mask
        image_patch = self.image
        mask_patch = self.mask

        # Normalize image and convert mask to binary if necessary
        image_patch = np.transpose(image_patch, (1, 2, 0))  # Convert [C, H, W] to [H, W, C]
        mask_patch = (mask_patch > 0).astype(np.uint8)  # Optional: Binarize mask

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image_patch, mask=mask_patch)
            image_patch = augmented["image"]
            mask_patch = augmented["mask"]

        return image_patch, mask_patch


# class SlumSegmentationDataset(Dataset):
#     def __init__(self, image_path, mask_path, transform=None):
#         self.image_path = image_path
#         self.mask_path = mask_path
#         self.transform = transform

#         # Load the entire image and mask at initialization
#         with rasterio.open(self.image_path) as src:
#             self.image = src.read()  # [Channels, Height, Width]
#             self.image_meta = src.meta

#         with rasterio.open(self.mask_path) as src:
#             self.mask = src.read(1)  # Single-channel mask
#             self.mask_meta = src.meta

#     def __len__(self):
#         # We are working with single images, so len = 1
#         return 1

#     def __getitem__(self, idx):
#         # Transpose the image to [H, W, C] for augmentations
#         image = np.transpose(self.image, (1, 2, 0))  # [H, W, C]
#         mask = self.mask  # [H, W]

#         # Normalize the image and mask
#         mask = (mask > 0).astype(np.uint8)  # Optional: Binarize the mask

#         # Apply augmentations if provided
#         if self.transform:
#             augmented = self.transform(image=image, mask=mask)
#             image = augmented["image"]
#             mask = augmented["mask"]

#         return image, mask







