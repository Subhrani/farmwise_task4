"""
Model architecture for multi-task crop and growth-stage classification
"""

import torch
import torch.nn as nn
import torchvision.models as models

class MultiTaskCropStageModel(nn.Module):
    """Multi-task model for crop and growth-stage classification"""
    
    def __init__(self, num_crop_classes, num_stage_classes, backbone='resnet18', dropout_rate=0.5, freeze_backbone=True):
        """
        Args:
            num_crop_classes: Number of crop classes
            num_stage_classes: Number of growth-stage classes
            backbone: Pretrained backbone architecture
            dropout_rate: Dropout rate for classification heads
            freeze_backbone: Whether to freeze backbone initially
        """
        super(MultiTaskCropStageModel, self).__init__()
        
        self.num_crop_classes = num_crop_classes
        self.num_stage_classes = num_stage_classes
        self.freeze_backbone = freeze_backbone
        
        # Load pretrained backbone
        try:
            if backbone == 'resnet18':
                self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                num_features = self.backbone.fc.in_features
            elif backbone == 'resnet34':
                self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
                num_features = self.backbone.fc.in_features
            elif backbone == 'mobilenet_v2':
                self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
                num_features = self.backbone.classifier[-1].in_features
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights with new API, trying legacy API: {e}")
            # Fallback to legacy API for older torchvision versions
            if backbone == 'resnet18':
                self.backbone = models.resnet18(pretrained=True)
                num_features = self.backbone.fc.in_features
            elif backbone == 'resnet34':
                self.backbone = models.resnet34(pretrained=True)
                num_features = self.backbone.fc.in_features
            elif backbone == 'mobilenet_v2':
                self.backbone = models.mobilenet_v2(pretrained=True)
                num_features = self.backbone.classifier[-1].in_features
        
        # Remove the original classification layer
        if backbone.startswith('resnet'):
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif backbone == 'mobilenet_v2':
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Shared feature representation
        self.shared_features = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Crop classification head
        self.crop_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_crop_classes)
        )
        
        # Growth-stage classification head
        self.stage_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_stage_classes)
        )
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input images (batch_size, 3, 224, 224)
        Returns:
            crop_logits: (batch_size, num_crop_classes)
            stage_logits: (batch_size, num_stage_classes)
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Shared feature representation
        shared = self.shared_features(features)
        
        # Crop and stage predictions
        crop_logits = self.crop_head(shared)
        stage_logits = self.stage_head(shared)
        
        return crop_logits, stage_logits
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen for fine-tuning")

def create_model(config):
    """Create model with configuration"""
    model = MultiTaskCropStageModel(
        num_crop_classes=config.NUM_CROP_CLASSES,
        num_stage_classes=config.NUM_STAGE_CLASSES,
        backbone=config.BACKBONE,
        dropout_rate=config.DROPOUT_RATE,
        freeze_backbone=config.FREEZE_BACKBONE
    )
    
    model = model.to(config.DEVICE)
    
    return model