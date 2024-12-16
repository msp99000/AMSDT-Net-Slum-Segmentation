from torch.utils.data import DataLoader
import os
from sklearn.model_selection import train_test_split
from dataset import SlumSegmentationDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_loaders(image_path, mask_path, batch_size, train_transform, val_transform, 
                num_workers=4, pin_memory=True, input_channels=10):
    
    train_dataset = SlumSegmentationDataset(
        image_path=image_path,
        mask_path=mask_path,
        transform=train_transform,
        input_channels=input_channels  # Pass input_channels here
    )

    val_dataset = SlumSegmentationDataset(
        image_path=image_path,
        mask_path=mask_path,
        transform=val_transform,
        input_channels=input_channels  # Pass input_channels here
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



def get_training_augmentation():
    return A.Compose([
        A.Resize(height=256, width=256),  # Resize to a standard size
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # Adjust to your dataset's mean/std
        ToTensorV2(),
    ])


def get_validation_augmentation():
    return A.Compose([
        A.Resize(height=256, width=256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])



# def get_training_augmentation(target_height, target_width):
#     return A.Compose([
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
#         A.CenterCrop(height=target_height, width=target_width, always_apply=True),
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ToTensorV2(),
#     ])

