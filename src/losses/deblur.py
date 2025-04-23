import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=[2, 7, 12, 21, 30]):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        self.features = vgg.features
        self.layers = layers
        self._freeze_params()
        
    def _freeze_params(self):
        for param in self.features.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in self.layers:
                results.append(x)
        return results

class MotionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
    def forward(self, pred, target):
        # Convert to grayscale
        pred_gray = 0.299 * pred[:, 0] + 0.587 * pred[:, 1] + 0.114 * pred[:, 2]
        target_gray = 0.299 * target[:, 0] + 0.587 * target[:, 1] + 0.114 * target[:, 2]
        
        # Compute gradients
        pred_grad_x = F.conv2d(pred_gray.unsqueeze(1), self.sobel_x.to(pred.device), padding=1)
        pred_grad_y = F.conv2d(pred_gray.unsqueeze(1), self.sobel_y.to(pred.device), padding=1)
        target_grad_x = F.conv2d(target_gray.unsqueeze(1), self.sobel_x.to(target.device), padding=1)
        target_grad_y = F.conv2d(target_gray.unsqueeze(1), self.sobel_y.to(target.device), padding=1)
        
        # Compute gradient magnitude
        pred_grad_mag = torch.sqrt(pred_grad_x**2 + pred_grad_y**2)
        target_grad_mag = torch.sqrt(target_grad_x**2 + target_grad_y**2)
        
        # Compute motion loss
        motion_loss = F.l1_loss(pred_grad_mag, target_grad_mag)
        
        return motion_loss

class DeblurLoss(nn.Module):
    def __init__(self, l1_weight=1.0, perceptual_weight=0.1, motion_weight=0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.motion_weight = motion_weight
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss()
        self.motion_loss = MotionLoss()
        
    def forward(self, pred, target):
        # L1 Loss
        l1_loss = self.l1_loss(pred, target)
        
        # Perceptual Loss
        if self.perceptual_weight > 0:
            # Normalize images for VGG
            pred_features = self.perceptual_loss((pred + 1) / 2)
            target_features = self.perceptual_loss((target + 1) / 2)
            
            perceptual_loss = 0
            for pred_feat, target_feat in zip(pred_features, target_features):
                perceptual_loss += F.l1_loss(pred_feat, target_feat)
        else:
            perceptual_loss = 0
            
        # Motion Loss
        if self.motion_weight > 0:
            motion_loss = self.motion_loss(pred, target)
        else:
            motion_loss = 0
            
        # Total Loss
        total_loss = self.l1_weight * l1_loss + \
                    self.perceptual_weight * perceptual_loss + \
                    self.motion_weight * motion_loss
        
        return {
            'total_loss': total_loss,
            'l1_loss': l1_loss,
            'perceptual_loss': perceptual_loss,
            'motion_loss': motion_loss
        } 