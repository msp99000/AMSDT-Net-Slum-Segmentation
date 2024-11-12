from torch.utils.data import DataLoader
import os
from sklearn.model_selection import train_test_split
from dataset import SlumSegmentationDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_loaders(image_dir, mask_dir, batch_size, train_transform, val_transform, num_workers=4, pin_memory=True):

    train_dataset = SlumSegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_dataset = SlumSegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def get_training_augmentation():
    return A.Compose([
        A.RandomCrop(height=256, width=256, always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_validation_augmentation():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])