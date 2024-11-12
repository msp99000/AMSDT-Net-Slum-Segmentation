from torch import nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
    
class CombinedLoss(nn.Module):
    """Combined loss using CrossEntropy and Dice Loss.
    The CombinedLoss class creates a weighted sum of Binary Cross-Entropy (BCE) and Dice Loss, which provides a good balance between pixel-wise accuracy and overall segmentation quality.
    """
    def __init__(self, alpha=0.5):
        """
        The alpha parameter controls the balance between the two losses, default is 0.5:

When alpha = 1, it's equivalent to using only BCE loss
When alpha = 0, it's equivalent to using only Dice loss
Values between 0 and 1 create a weighted combination of both
        """
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.bce_loss = smp.losses.SoftBCEWithLogitsLoss()#nn.BCEWithLogitsLoss()
        self.dice_loss = smp.losses.DiceLoss("binary", from_logits=True)

    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.alpha * bce + (1 - self.alpha) * dice

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        intersection = (pred * target).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()