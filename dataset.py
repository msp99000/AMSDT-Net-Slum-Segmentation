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


# class SlumSegmentationDataset(Dataset):
#     def __init__(self, image_dir, mask_dir, transform=None, patch_size=4, stride=4):
#         self.image_dir = image_dir
#         self.mask_dir = mask_dir
#         self.transform = transform
#         self.patch_size = patch_size
#         self.stride = stride
#         self.images = sorted(os.listdir(image_dir))
#         self.masks = sorted(os.listdir(mask_dir))  # Ensure masks are sorted consistently
#         self.image_to_mask = {img: f"mask_{img.split('_')[1]}" for img in self.images}

#         # Precompute patch positions for all images
#         self.patches = []
#         for img_idx, img_name in enumerate(self.images):
#             img_path = os.path.join(self.image_dir, img_name)
#             mask_name = self.image_to_mask.get(img_name, None)
#             if mask_name is None:
#                 raise ValueError(f"No corresponding mask found for image: {img_name}")
#             mask_path = os.path.join(self.mask_dir, mask_name)

#             image = cv2.imread(img_path)
#             mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

#             h, w = image.shape[:2]
#             for i in range(0, h - patch_size + 1, stride):
#                 for j in range(0, w - patch_size + 1, stride):
#                     self.patches.append((img_idx, i, j))

#     def __len__(self):
#         return len(self.patches)

#     def __getitem__(self, idx):
#         # Retrieve image and patch positions
#         img_idx, patch_x, patch_y = self.patches[idx]
#         img_name = self.images[img_idx]
#         mask_name = self.image_to_mask.get(img_name, None)

#         img_path = os.path.join(self.image_dir, img_name)
#         mask_path = os.path.join(self.mask_dir, mask_name)

#         image = cv2.imread(img_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

#         # Extract patch
#         patch_img = image[patch_x:patch_x + self.patch_size, patch_y:patch_y + self.patch_size]
#         patch_mask = mask[patch_x:patch_x + self.patch_size, patch_y:patch_y + self.patch_size]

#         # Apply transformation
#         if self.transform:
#             augmented = self.transform(image=patch_img, mask=patch_mask)
#             patch_img = augmented['image']
#             patch_mask = augmented['mask']

#         return patch_img, patch_mask

# class SlumSegmentationDataset(Dataset):
#     def __init__(self, image_dir, mask_dir, transform=None):
#         self.image_dir = image_dir
#         self.mask_dir = mask_dir
#         self.transform = transform
#         self.images = sorted(os.listdir(image_dir))
#         self.masks = sorted(os.listdir(mask_dir))  # Ensure masks are sorted consistently
#         self.image_to_mask = {img: f"mask_{img.split('_')[1]}" for img in self.images}

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         img_name = self.images[idx]
#         mask_name = self.image_to_mask.get(img_name, None)
#         if mask_name is None:
#             raise ValueError(f"No corresponding mask found for image: {img_name}")

#         img_path = os.path.join(self.image_dir, img_name)
#         mask_path = os.path.join(self.mask_dir, mask_name)

#         image = cv2.imread(img_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

#         if self.transform:
#             augmented = self.transform(image=image, mask=mask)
#             image = augmented['image']
#             mask = augmented['mask']

#         return image, mask


