"""
Dataset module for loading crop and growth-stage classification data
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import csv
from pathlib import Path
import numpy as np

class CropStageDataset(Dataset):
    """Dataset for multi-task crop and growth-stage classification"""
    
    def __init__(self, csv_path, images_dir, transform=None):
        """
        Args:
            csv_path: Path to CSV file with image paths and labels
            images_dir: Directory containing the images
            transform: Optional transform to apply to images
        """
        self.data = []
        self.images_dir = Path(images_dir)
        self.transform = transform
        
        # Read CSV
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row)
        
        # Create label mappings
        crop_set = set()
        stage_set = set()
        for row in self.data:
            crop_set.add(row['crop'])
            stage_set.add(row['growth_stage'])
        
        self.crop_classes = sorted(crop_set)
        self.stage_classes = sorted(stage_set)
        
        self.crop_to_idx = {crop: idx for idx, crop in enumerate(self.crop_classes)}
        self.stage_to_idx = {stage: idx for idx, stage in enumerate(self.stage_classes)}
        
        print(f"Loaded {len(self.data)} samples")
        print(f"Crop classes: {len(self.crop_classes)}")
        print(f"Stage classes: {len(self.stage_classes)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        
        # Load image
        image_path = self.images_dir / Path(row['image_file']).name
        image = Image.open(str(image_path)).convert('RGB')
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        crop_label = self.crop_to_idx[row['crop']]
        stage_label = self.stage_to_idx[row['growth_stage']]
        
        return {
            'image': image,
            'crop_label': torch.tensor(crop_label, dtype=torch.long),
            'stage_label': torch.tensor(stage_label, dtype=torch.long),
            'image_file': row['image_file']
        }

def get_transforms(config):
    """Get training and validation/test transforms"""
    
    # Training transform with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=config.TRAIN_HORIZONTAL_FLIP_PROB),
        transforms.RandomRotation(config.TRAIN_ROTATION_DEGREES),
        transforms.ColorJitter(brightness=config.TRAIN_BRIGHTNESS, contrast=config.TRAIN_CONTRAST),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    # Validation/test transform (deterministic)
    val_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_transform

def calculate_class_weights(csv_path, num_classes, label_column='crop'):
    """Calculate class weights for imbalanced dataset"""
    class_counts = {}
    total_samples = 0
    
    # Read CSV and count classes
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row[label_column]
            class_counts[label] = class_counts.get(label, 0) + 1
            total_samples += 1
    
    # Calculate weights (inverse frequency)
    class_weights = {}
    for class_name in sorted(class_counts.keys()):
        count = class_counts[class_name]
        weight = total_samples / (num_classes * count)
        class_weights[class_name] = weight
    
    return class_weights

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    
    # Get transforms
    train_transform, val_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = CropStageDataset(config.TRAIN_CSV, config.IMAGES_DIR, transform=train_transform)
    val_dataset = CropStageDataset(config.VAL_CSV, config.IMAGES_DIR, transform=val_transform)
    test_dataset = CropStageDataset(config.TEST_CSV, config.IMAGES_DIR, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    # Calculate class weights for crop and stage separately
    crop_class_weights = calculate_class_weights(config.TRAIN_CSV, config.NUM_CROP_CLASSES, 'crop')
    stage_class_weights = calculate_class_weights(config.TRAIN_CSV, config.NUM_STAGE_CLASSES, 'growth_stage')
    
    # Convert to tensors
    crop_weight_list = [crop_class_weights[crop] for crop in train_dataset.crop_classes]
    stage_weight_list = [stage_class_weights[stage] for stage in train_dataset.stage_classes]
    
    crop_weights_tensor = torch.tensor(crop_weight_list, dtype=torch.float32).to(config.DEVICE)
    stage_weights_tensor = torch.tensor(stage_weight_list, dtype=torch.float32).to(config.DEVICE)
    
    return train_loader, val_loader, test_loader, crop_weights_tensor, stage_weights_tensor, train_dataset.crop_classes, train_dataset.stage_classes