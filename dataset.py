from torch.utils.data import Dataset
import os
import cv2
import numpy as np 

class SlumSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, patch_size=4, stride=4):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))  # Ensure masks are sorted consistently
        self.image_to_mask = {img: f"mask_{img.split('_')[1]}" for img in self.images}

        # Precompute patch positions for all images
        self.patches = []
        for img_idx, img_name in enumerate(self.images):
            img_path = os.path.join(self.image_dir, img_name)
            mask_name = self.image_to_mask.get(img_name, None)
            if mask_name is None:
                raise ValueError(f"No corresponding mask found for image: {img_name}")
            mask_path = os.path.join(self.mask_dir, mask_name)

            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            h, w = image.shape[:2]
            for i in range(0, h - patch_size + 1, stride):
                for j in range(0, w - patch_size + 1, stride):
                    self.patches.append((img_idx, i, j))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        # Retrieve image and patch positions
        img_idx, patch_x, patch_y = self.patches[idx]
        img_name = self.images[img_idx]
        mask_name = self.image_to_mask.get(img_name, None)

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Extract patch
        patch_img = image[patch_x:patch_x + self.patch_size, patch_y:patch_y + self.patch_size]
        patch_mask = mask[patch_x:patch_x + self.patch_size, patch_y:patch_y + self.patch_size]

        # Apply transformation
        if self.transform:
            augmented = self.transform(image=patch_img, mask=patch_mask)
            patch_img = augmented['image']
            patch_mask = augmented['mask']

        return patch_img, patch_mask

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


