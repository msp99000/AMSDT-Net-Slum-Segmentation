from torch.utils.data import DataLoader
import os
from sklearn.model_selection import train_test_split
from dataset import SlumSegmentationDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# def get_loaders(image_dir, mask_dir, batch_size, train_transform, val_transform, patch_size=128, stride=128):
#     train_dataset = SlumSegmentationDataset(
#         image_dir=image_dir,
#         mask_dir=mask_dir,
#         transform=train_transform,
#         patch_size=patch_size,
#         stride=stride
#     )
#     val_dataset = SlumSegmentationDataset(
#         image_dir=image_dir,
#         mask_dir=mask_dir,
#         transform=val_transform,
#         patch_size=patch_size,
#         stride=stride
#     )

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         num_workers=4,
#         shuffle=True
#     )
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         num_workers=4,
#         shuffle=False
#     )

#     return train_loader, val_loader


def get_loaders(image_path, mask_path, batch_size, train_transform, val_transform, 
                patch_size=None, stride=None, num_workers=4, pin_memory=True):
    # Create train and validation datasets
    train_dataset = SlumSegmentationDataset(
        image_path=image_path,
        mask_path=mask_path,
        transform=train_transform,
        patch_size=patch_size,
        stride=stride
    )

    val_dataset = SlumSegmentationDataset(
        image_path=image_path,
        mask_path=mask_path,
        transform=val_transform,
        patch_size=patch_size,
        stride=stride
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, val_loader


def get_training_augmentation(target_height, target_width):
    return A.Compose([
        A.RandomCrop(height=target_height, width=target_width, always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Update if `.tif` data requires different normalization
        ToTensorV2(),
    ])


def get_validation_augmentation(target_height, target_width):
    return A.Compose([
        A.CenterCrop(height=target_height, width=target_width, always_apply=True),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# def get_training_augmentation(target_height, target_width):
#     return A.Compose([
#         A.Resize(height=target_height, width=target_width),  # Resize patches to consistent size
#         # Ensure the crop size matches or is smaller than the patch size
#         A.RandomCrop(height=target_height, width=target_width, always_apply=True),
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.RandomRotate90(p=0.5),
#         A.RandomBrightnessContrast(p=0.2),
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ToTensorV2(),
#     ])

# def get_validation_augmentation(target_height, target_width):
#     return A.Compose([
#         A.Resize(height=target_height, width=target_width),  # Resize patches to consistent size
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ToTensorV2(),
#     ])

