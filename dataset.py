import rasterio
from torch.utils.data import Dataset
import os
import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations as A


class SlumSegmentationDataset(Dataset):
    def __init__(self, image_path, mask_path, transform=None, patch_size=None, stride=None):
        self.image_path = image_path  # Single .tif file path for the image
        self.mask_path = mask_path    # Single .tif file path for the mask
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride

        # Open image and mask once
        with rasterio.open(self.image_path) as src:
            self.image = src.read()  # Shape: [channels, height, width]
            self.image_meta = src.meta

        with rasterio.open(self.mask_path) as src:
            self.mask = src.read(1)  # Shape: [height, width] (assuming mask is single-channel)
            self.mask_meta = src.meta

    def __len__(self):
        if self.patch_size and self.stride:
            height, width = self.image.shape[1:]
            num_patches = (
                (height - self.patch_size) // self.stride + 1
            ) * (
                (width - self.patch_size) // self.stride + 1
            )
            return num_patches
        return 1

    def __getitem__(self, idx):
        # Patch extraction
        if self.patch_size and self.stride:
            height, width = self.image.shape[1:]
            num_patches_per_row = (width - self.patch_size) // self.stride + 1

            patch_y = (idx // num_patches_per_row) * self.stride
            patch_x = (idx % num_patches_per_row) * self.stride

            image_patch = self.image[:, patch_y:patch_y + self.patch_size, patch_x:patch_x + self.patch_size]
            mask_patch = self.mask[patch_y:patch_y + self.patch_size, patch_x:patch_x + self.patch_size]
        else:
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




