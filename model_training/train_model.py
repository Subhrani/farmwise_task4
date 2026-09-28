"""
Training script for multi-task crop and growth-stage classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
import csv
import config
from dataset import create_dataloaders
from model import create_model

def train_epoch(model, dataloader, crop_criterion, stage_criterion, optimizer, device, crop_weight, stage_weight):
    """Train for one epoch"""
    model.train()
    
    running_loss = 0.0
    crop_correct = 0
    stage_correct = 0
    total = 0
    
    for batch in dataloader:
        images = batch['image'].to(device)
        crop_labels = batch['crop_label'].to(device)
        stage_labels = batch['stage_label'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        crop_logits, stage_logits = model(images)
        
        # Calculate loss
        crop_loss = crop_criterion(crop_logits, crop_labels)
        stage_loss = stage_criterion(stage_logits, stage_labels)
        
        # Combined loss
        total_loss = crop_weight * crop_loss + stage_weight * stage_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += total_loss.item()
        
        _, crop_pred = torch.max(crop_logits, 1)
        _, stage_pred = torch.max(stage_logits, 1)
        
        crop_correct += (crop_pred == crop_labels).sum().item()
        stage_correct += (stage_pred == stage_labels).sum().item()
        total += crop_labels.size(0)
    
    epoch_loss = running_loss / len(dataloader)
    crop_acc = crop_correct / total
    stage_acc = stage_correct / total
    
    return epoch_loss, crop_acc, stage_acc

def validate(model, dataloader, crop_criterion, stage_criterion, device, crop_weight, stage_weight):
    """Validate the model"""
    model.eval()
    
    running_loss = 0.0
    crop_correct = 0
    stage_correct = 0
    total = 0
    
    all_crop_preds = []
    all_crop_labels = []
    all_stage_preds = []
    all_stage_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            crop_labels = batch['crop_label'].to(device)
            stage_labels = batch['stage_label'].to(device)
            
            # Forward pass
            crop_logits, stage_logits = model(images)
            
            # Calculate loss
            crop_loss = crop_criterion(crop_logits, crop_labels)
            stage_loss = stage_criterion(stage_logits, stage_labels)
            
            # Combined loss
            total_loss = crop_weight * crop_loss + stage_weight * stage_loss
            
            # Statistics
            running_loss += total_loss.item()
            
            _, crop_pred = torch.max(crop_logits, 1)
            _, stage_pred = torch.max(stage_logits, 1)
            
            crop_correct += (crop_pred == crop_labels).sum().item()
            stage_correct += (stage_pred == stage_labels).sum().item()
            total += crop_labels.size(0)
            
            all_crop_preds.extend(crop_pred.cpu().numpy())
            all_crop_labels.extend(crop_labels.cpu().numpy())
            all_stage_preds.extend(stage_pred.cpu().numpy())
            all_stage_labels.extend(stage_labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    crop_acc = crop_correct / total
    stage_acc = stage_correct / total
    
    return epoch_loss, crop_acc, stage_acc, all_crop_preds, all_crop_labels, all_stage_preds, all_stage_labels

def train_model(config):
    """Main training function"""
    print("=" * 80)
    print("MULTI-TASK CROP + GROWTH-STAGE CLASSIFICATION TRAINING")
    print("=" * 80)
    print()
    
    # Set random seed
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    print(f"Device: {config.DEVICE}")
    print(f"Backbone: {config.BACKBONE}")
    print(f"Image size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Crop weight: {config.CROP_WEIGHT}")
    print(f"Stage weight: {config.STAGE_WEIGHT}")
    print()
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader, crop_weights, stage_weights, crop_classes, stage_classes = create_dataloaders(config)
    
    # Use subset of training data for faster baseline training
    from torch.utils.data import Subset
    subset_size = 500  # Use only 500 training samples for baseline
    train_indices = list(range(min(subset_size, len(train_loader.dataset))))
    train_subset = Subset(train_loader.dataset, train_indices)
    train_loader = DataLoader(
        train_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    print(f"Using subset of {len(train_loader.dataset)} training samples for baseline")
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Crop weights: {crop_weights.tolist()}")
    print(f"Stage weights: {stage_weights.tolist()}")
    print()
    
    # Save class names
    class_names = {
        'crop_classes': crop_classes,
        'stage_classes': stage_classes
    }
    
    with open(config.ARTIFACTS_DIR / "class_names.json", 'w') as f:
        json.dump(class_names, f, indent=2)
    
    # Save training configuration
    training_config = {
        'random_seed': config.RANDOM_SEED,
        'backbone': config.BACKBONE,
        'image_size': config.IMAGE_SIZE,
        'batch_size': config.BATCH_SIZE,
        'num_epochs': config.NUM_EPOCHS,
        'learning_rate': config.LEARNING_RATE,
        'weight_decay': config.WEIGHT_DECAY,
        'momentum': config.MOMENTUM,
        'crop_weight': config.CROP_WEIGHT,
        'stage_weight': config.STAGE_WEIGHT,
        'dropout_rate': config.DROPOUT_RATE,
        'freeze_backbone': config.FREEZE_BACKBONE,
        'freeze_backbone_epochs': config.FREEZE_BACKBONE_EPOCHS,
        'train_samples': len(train_loader.dataset),
        'val_samples': len(val_loader.dataset),
        'test_samples': len(test_loader.dataset),
        'num_crop_classes': config.NUM_CROP_CLASSES,
        'num_stage_classes': config.NUM_STAGE_CLASSES,
        'crop_weights': crop_weights.tolist(),
        'stage_weights': stage_weights.tolist()
    }
    
    with open(config.ARTIFACTS_DIR / "training_config.json", 'w') as f:
        json.dump(training_config, f, indent=2)
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    
    # Loss functions with class weights
    crop_criterion = nn.CrossEntropyLoss(weight=crop_weights)
    stage_criterion = nn.CrossEntropyLoss(weight=stage_weights)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = StepLR(
        optimizer,
        step_size=config.LR_SCHEDULER_STEP_SIZE,
        gamma=config.LR_SCHEDULER_GAMMA
    )
    
    # Training history
    history = {
        'epoch': [],
        'train_loss': [],
        'train_crop_acc': [],
        'train_stage_acc': [],
        'val_loss': [],
        'val_crop_acc': [],
        'val_stage_acc': []
    }
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(config.NUM_EPOCHS):
        # Train
        train_loss, train_crop_acc, train_stage_acc = train_epoch(
            model, train_loader, crop_criterion, stage_criterion, optimizer, config.DEVICE,
            config.CROP_WEIGHT, config.STAGE_WEIGHT
        )
        
        # Validate
        val_loss, val_crop_acc, val_stage_acc, _, _, _, _ = validate(
            model, val_loader, crop_criterion, stage_criterion, config.DEVICE,
            config.CROP_WEIGHT, config.STAGE_WEIGHT
        )
        
        # Update learning rate
        scheduler.step()
        
        # Record history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_crop_acc'].append(train_crop_acc)
        history['train_stage_acc'].append(train_stage_acc)
        history['val_loss'].append(val_loss)
        history['val_crop_acc'].append(val_crop_acc)
        history['val_stage_acc'].append(val_stage_acc)
        
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}, Crop Acc: {train_crop_acc:.4f}, Stage Acc: {train_stage_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Crop Acc: {val_crop_acc:.4f}, Stage Acc: {val_stage_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss - config.MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), config.CHECKPOINT_DIR / "best_model.pth")
            print(f"  Model saved (new best validation loss)")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/60:.1f} minutes")
    
    # Save training history as JSON
    with open(config.METRICS_DIR / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\nTraining completed. Best model saved to best_model.pth")

if __name__ == "__main__":
    train_model(config)