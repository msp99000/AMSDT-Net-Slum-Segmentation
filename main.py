import os
import numpy as np
import torch
import torch.amp
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from architecture.model import AMMTU_Net
from architecture.loss import CombinedLoss
from utils import get_loaders

def train_fn(loader, model, optimizer, loss_fn, scaler, device):
    model.train()
    total_loss = 0

    for batch_idx, (data, targets) in enumerate(tqdm(loader)):
        data = data.to(device)
        targets = targets.to(device)

        # forward
        with torch.amp.autocast("cuda"):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)

def eval_fn(loader, model, loss_fn, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data, targets in tqdm(loader):
            data = data.to(device)
            targets = targets.to(device)

            predictions = model(data)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()

    return total_loss / len(loader)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np
from sklearn.metrics import jaccard_score
from captum.attr import IntegratedGradients, visualization as viz

# Assuming we have our AMMTU_Net, dataset, and training functions defined as before

def iou_score(pred, target):
    pred = (pred > 0.5).float()
    return jaccard_score(target.cpu().numpy().flatten(), pred.cpu().numpy().flatten())

class BaselineUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(BaselineUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.unet = models.segmentation.fcn_resnet50(pretrained=False, num_classes=n_classes)
        self.unet.backbone.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        return self.unet(x)['out']

def ablation_study(model, train_loader, val_loader, device, epochs=50):
    results = {}
    
    # Full model
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    full_model_iou = train_and_evaluate(model, train_loader, val_loader, optimizer, device, epochs)
    results['Full Model'] = full_model_iou
    
    # Without Multi-Scale Feature Extraction
    model.multi_scale = nn.Identity()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    no_multiscale_iou = train_and_evaluate(model, train_loader, val_loader, optimizer, device, epochs)
    results['No Multi-Scale'] = no_multiscale_iou
    
    # Without Transformer Encoder
    model.transformer_encoder = nn.Identity()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    no_transformer_iou = train_and_evaluate(model, train_loader, val_loader, optimizer, device, epochs)
    results['No Transformer'] = no_transformer_iou
    
    # Without Boundary Refinement
    model.boundary_refinement = nn.Identity()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    no_boundary_iou = train_and_evaluate(model, train_loader, val_loader, optimizer, device, epochs)
    results['No Boundary Refinement'] = no_boundary_iou
    
    return results

def feature_study(model, sample_image, target, device):
    model.eval()
    ig = IntegratedGradients(model)
    attr = ig.attribute(sample_image.unsqueeze(0).to(device), target=target, n_steps=200)
    
    attr_vis = viz.visualize_image_attr(
        np.transpose(attr.squeeze().cpu().detach().numpy(), (1,2,0)),
        np.transpose(sample_image.squeeze().cpu().numpy(), (1,2,0)),
        method='heat_map',
        sign='positive',
        show_colorbar=True,
        title='Integrated Gradients'
    )
    
    return attr_vis

def compare_baselines(ammtu_net, unet, train_loader, val_loader, device, epochs=50):
    results = {}
    
    # AMMTU-Net
    optimizer = optim.Adam(ammtu_net.parameters(), lr=1e-4)
    ammtu_iou = train_and_evaluate(ammtu_net, train_loader, val_loader, optimizer, device, epochs)
    results['AMMTU-Net'] = ammtu_iou
    
    # U-Net
    optimizer = optim.Adam(unet.parameters(), lr=1e-4)
    unet_iou = train_and_evaluate(unet, train_loader, val_loader, optimizer, device, epochs)
    results['U-Net'] = unet_iou
    
    return results

def main():
    # Hyperparameters
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    NUM_WORKERS = 4
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    PIN_MEMORY = True
    LOAD_MODEL = False

    # Directories
    IMAGE_DIR = "path/to/image/directory"
    MASK_DIR = "path/to/mask/directory"

    # Ablation Studies
    ablation_results = ablation_study(model, train_loader, val_loader, DEVICE)
    print("Ablation Study Results:")
    for key, value in ablation_results.items():
        print(f"{key}: IoU = {value:.4f}")
    
    # Feature Study
    sample_image, sample_target = next(iter(val_loader))
    feature_visualization = feature_study(model, sample_image[0], sample_target[0], DEVICE)
    feature_visualization.savefig('feature_study.png')
    
    # Baseline Comparison
    unet = BaselineUNet(n_channels=3, n_classes=1).to(DEVICE)
    baseline_results = compare_baselines(model, unet, train_loader, val_loader, DEVICE)
    print("Baseline Comparison Results:")
    for key, value in baseline_results.items():
        print(f"{key}: IoU = {value:.4f}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
    ])

    # Model
    model = AMMTU_Net(satellite_channels=3, num_classes=1).to(DEVICE)
    loss_fn = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        IMAGE_DIR,
        MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    scaler = torch.cuda.amp.GradScaler()
    
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, DEVICE)
        val_loss = eval_fn(val_loader, model, loss_fn, DEVICE)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        # Save model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved best model")

if __name__ == "__main__":
    main()