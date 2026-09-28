"""
Evaluation script for multi-task crop and growth-stage classification
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import csv
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import config
from dataset import create_dataloaders
from model import create_model

def calculate_metrics(y_true, y_pred, num_classes):
    """Calculate precision, recall, F1 manually"""
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate per-class metrics
    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    per_class_support = []
    
    for c in range(num_classes):
        true_positives = np.sum((y_true == c) & (y_pred == c))
        false_positives = np.sum((y_true != c) & (y_pred == c))
        false_negatives = np.sum((y_true == c) & (y_pred != c))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = np.sum(y_true == c)
        
        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_f1.append(f1)
        per_class_support.append(support)
    
    # Calculate macro averages
    macro_precision = np.mean(per_class_precision)
    macro_recall = np.mean(per_class_recall)
    macro_f1 = np.mean(per_class_f1)
    
    # Calculate weighted averages
    total_support = sum(per_class_support)
    weighted_precision = sum(p * s for p, s in zip(per_class_precision, per_class_support)) / total_support if total_support > 0 else 0
    weighted_recall = sum(r * s for r, s in zip(per_class_recall, per_class_support)) / total_support if total_support > 0 else 0
    weighted_f1 = sum(f * s for f, s in zip(per_class_f1, per_class_support)) / total_support if total_support > 0 else 0
    
    return {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'per_class_support': per_class_support
    }

def evaluate_split(model, dataloader, crop_criterion, stage_criterion, device, crop_weight, stage_weight, split_name):
    """Evaluate model on a specific split"""
    model.eval()
    
    running_loss = 0.0
    crop_correct = 0
    stage_correct = 0
    joint_correct = 0
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
            joint_correct += ((crop_pred == crop_labels) & (stage_pred == stage_labels)).sum().item()
            total += crop_labels.size(0)
            
            all_crop_preds.extend(crop_pred.cpu().numpy())
            all_crop_labels.extend(crop_labels.cpu().numpy())
            all_stage_preds.extend(stage_pred.cpu().numpy())
            all_stage_labels.extend(stage_labels.cpu().numpy())
    
    loss = running_loss / len(dataloader)
    crop_acc = crop_correct / total
    stage_acc = stage_correct / total
    joint_acc = joint_correct / total
    
    print(f"\n{split_name} Results:")
    print(f"  Loss: {loss:.4f}")
    print(f"  Crop Accuracy: {crop_acc:.4f}")
    print(f"  Stage Accuracy: {stage_acc:.4f}")
    print(f"  Joint Crop+Stage Accuracy: {joint_acc:.4f}")
    
    return {
        'loss': loss,
        'crop_acc': crop_acc,
        'stage_acc': stage_acc,
        'joint_acc': joint_acc,
        'crop_preds': all_crop_preds,
        'crop_labels': all_crop_labels,
        'stage_preds': all_stage_preds,
        'stage_labels': all_stage_labels
    }

def calculate_metrics_with_names(y_true, y_pred, class_names):
    """Calculate detailed metrics for a classification task"""
    num_classes = len(class_names)
    metrics_data = calculate_metrics(y_true, y_pred, num_classes)
    
    # Build confusion matrix manually
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    
    metrics = {
        'macro_precision': metrics_data['macro_precision'],
        'macro_recall': metrics_data['macro_recall'],
        'macro_f1': metrics_data['macro_f1'],
        'weighted_precision': metrics_data['weighted_precision'],
        'weighted_recall': metrics_data['weighted_recall'],
        'weighted_f1': metrics_data['weighted_f1'],
        'per_class_metrics': {
            class_name: {
                'precision': p,
                'recall': r,
                'f1': f,
                'support': s
            }
            for class_name, p, r, f, s in zip(class_names, metrics_data['per_class_precision'], metrics_data['per_class_recall'], metrics_data['per_class_f1'], metrics_data['per_class_support'])
        },
        'confusion_matrix': cm.tolist()
    }
    
    return metrics

def plot_confusion_matrix(cm, class_names, output_path, title):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def evaluate_model(config):
    """Main evaluation function"""
    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    print()
    
    # Load class names
    with open(config.ARTIFACTS_DIR / "class_names.json", 'r') as f:
        class_names = json.load(f)
    
    crop_classes = class_names['crop_classes']
    stage_classes = class_names['stage_classes']
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader, crop_weights, stage_weights, _, _ = create_dataloaders(config)
    
    # Load model
    print("Loading trained model...")
    model = create_model(config)
    model.load_state_dict(torch.load(config.CHECKPOINT_DIR / "best_model.pth"))
    model = model.to(config.DEVICE)
    
    # Loss functions
    crop_criterion = nn.CrossEntropyLoss(weight=crop_weights)
    stage_criterion = nn.CrossEntropyLoss(weight=stage_weights)
    
    # Evaluate on all splits
    train_results = evaluate_split(
        model, train_loader, crop_criterion, stage_criterion, config.DEVICE,
        config.CROP_WEIGHT, config.STAGE_WEIGHT, "TRAIN"
    )
    
    val_results = evaluate_split(
        model, val_loader, crop_criterion, stage_criterion, config.DEVICE,
        config.CROP_WEIGHT, config.STAGE_WEIGHT, "VALIDATION"
    )
    
    test_results = evaluate_split(
        model, test_loader, crop_criterion, stage_criterion, config.DEVICE,
        config.CROP_WEIGHT, config.STAGE_WEIGHT, "TEST"
    )
    
    # Calculate detailed metrics for each split
    print("\nCalculating detailed metrics...")
    
    # Crop metrics
    train_crop_metrics = calculate_metrics_with_names(train_results['crop_labels'], train_results['crop_preds'], crop_classes)
    val_crop_metrics = calculate_metrics_with_names(val_results['crop_labels'], val_results['crop_preds'], crop_classes)
    test_crop_metrics = calculate_metrics_with_names(test_results['crop_labels'], test_results['crop_preds'], crop_classes)
    
    # Stage metrics
    train_stage_metrics = calculate_metrics_with_names(train_results['stage_labels'], train_results['stage_preds'], stage_classes)
    val_stage_metrics = calculate_metrics_with_names(val_results['stage_labels'], val_results['stage_preds'], stage_classes)
    test_stage_metrics = calculate_metrics_with_names(test_results['stage_labels'], test_results['stage_preds'], stage_classes)
    
    # Save crop metrics as CSV
    with open(config.METRICS_DIR / "crop_metrics.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1'])
        writer.writerow(['train', train_results['crop_acc'], train_crop_metrics['macro_precision'], train_crop_metrics['macro_recall'], train_crop_metrics['macro_f1'], train_crop_metrics['weighted_f1']])
        writer.writerow(['validation', val_results['crop_acc'], val_crop_metrics['macro_precision'], val_crop_metrics['macro_recall'], val_crop_metrics['macro_f1'], val_crop_metrics['weighted_f1']])
        writer.writerow(['test', test_results['crop_acc'], test_crop_metrics['macro_precision'], test_crop_metrics['macro_recall'], test_crop_metrics['macro_f1'], test_crop_metrics['weighted_f1']])
    
    # Save stage metrics as CSV
    with open(config.METRICS_DIR / "stage_metrics.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1'])
        writer.writerow(['train', train_results['stage_acc'], train_stage_metrics['macro_precision'], train_stage_metrics['macro_recall'], train_stage_metrics['macro_f1'], train_stage_metrics['weighted_f1']])
        writer.writerow(['validation', val_results['stage_acc'], val_stage_metrics['macro_precision'], val_stage_metrics['macro_recall'], val_stage_metrics['macro_f1'], val_stage_metrics['weighted_f1']])
        writer.writerow(['test', test_results['stage_acc'], test_stage_metrics['macro_precision'], test_stage_metrics['macro_recall'], test_stage_metrics['macro_f1'], test_stage_metrics['weighted_f1']])
    
    # Save test metrics JSON
    test_metrics = {
        'crop': {
            'accuracy': test_results['crop_acc'],
            'macro_precision': test_crop_metrics['macro_precision'],
            'macro_recall': test_crop_metrics['macro_recall'],
            'macro_f1': test_crop_metrics['macro_f1'],
            'weighted_f1': test_crop_metrics['weighted_f1'],
            'per_class_metrics': test_crop_metrics['per_class_metrics']
        },
        'stage': {
            'accuracy': test_results['stage_acc'],
            'macro_precision': test_stage_metrics['macro_precision'],
            'macro_recall': test_stage_metrics['macro_recall'],
            'macro_f1': test_stage_metrics['macro_f1'],
            'weighted_f1': test_stage_metrics['weighted_f1'],
            'per_class_metrics': test_stage_metrics['per_class_metrics']
        },
        'joint': {
            'accuracy': test_results['joint_acc']
        }
    }
    
    with open(config.METRICS_DIR / "test_metrics.json", 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    # Plot confusion matrices
    print("Plotting confusion matrices...")
    plot_confusion_matrix(
        test_crop_metrics['confusion_matrix'],
        crop_classes,
        config.METRICS_DIR / "crop_confusion_matrix.png",
        "Crop Classification Confusion Matrix (Test)"
    )
    
    plot_confusion_matrix(
        test_stage_metrics['confusion_matrix'],
        stage_classes,
        config.METRICS_DIR / "stage_confusion_matrix.png",
        "Growth-Stage Classification Confusion Matrix (Test)"
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    print("\nCrop Classification:")
    print(f"  Train Accuracy: {train_results['crop_acc']:.4f}")
    print(f"  Validation Accuracy: {val_results['crop_acc']:.4f}")
    print(f"  Test Accuracy: {test_results['crop_acc']:.4f}")
    print(f"  Test Macro F1: {test_crop_metrics['macro_f1']:.4f}")
    
    print("\nGrowth-Stage Classification:")
    print(f"  Train Accuracy: {train_results['stage_acc']:.4f}")
    print(f"  Validation Accuracy: {val_results['stage_acc']:.4f}")
    print(f"  Test Accuracy: {test_results['stage_acc']:.4f}")
    print(f"  Test Macro F1: {test_stage_metrics['macro_f1']:.4f}")
    
    print("\nJoint Crop+Stage Accuracy:")
    print(f"  Train: {train_results['joint_acc']:.4f}")
    print(f"  Validation: {val_results['joint_acc']:.4f}")
    print(f"  Test: {test_results['joint_acc']:.4f}")
    
    print("\nSmall Class Limitations:")
    print("  Note: Classes with very few samples in test set may have unreliable metrics.")
    print("  See test_metrics.json for per-class details.")
    
    print("\nMetrics saved to metrics/ directory")

if __name__ == "__main__":
    evaluate_model(config)