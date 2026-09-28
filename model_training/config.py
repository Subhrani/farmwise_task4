"""
Configuration file for multi-task crop and growth-stage classification
"""

import torch
from pathlib import Path

# Paths
BASE_DIR = Path(r"C:\Users\Gayathri B\Documents\task4")
CLEAN_DATASET_DIR = BASE_DIR / "clean_dataset"
SPLITS_DIR = CLEAN_DATASET_DIR / "splits"
IMAGES_DIR = CLEAN_DATASET_DIR / "images"
OUTPUT_DIR = BASE_DIR / "model_training"

# Dataset paths
TRAIN_CSV = SPLITS_DIR / "train.csv"
VAL_CSV = SPLITS_DIR / "validation.csv"
TEST_CSV = SPLITS_DIR / "test.csv"

# Model output paths
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
METRICS_DIR = OUTPUT_DIR / "metrics"
ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"

# Random seed for reproducibility
RANDOM_SEED = 42

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing
IMAGE_SIZE = 224  # Standard size for most pretrained models
MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
STD = [0.229, 0.224, 0.225]   # ImageNet std

# Training hyperparameters
BATCH_SIZE = 16  # Reduced for faster training
NUM_EPOCHS = 20  # Reduced for faster baseline
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9

# Learning rate scheduler
LR_SCHEDULER_STEP_SIZE = 10
LR_SCHEDULER_GAMMA = 0.1

# Early stopping
PATIENCE = 5
MIN_DELTA = 0.001

# Loss weights
CROP_WEIGHT = 1.0
STAGE_WEIGHT = 1.0

# Model architecture
BACKBONE = "resnet18"  # Lightweight pretrained backbone
NUM_CROP_CLASSES = 12
NUM_STAGE_CLASSES = 5
DROPOUT_RATE = 0.5

# Freezing backbone
FREEZE_BACKBONE = False  # Train entire backbone for simplicity
FREEZE_BACKBONE_EPOCHS = 0

# Data augmentation
TRAIN_HORIZONTAL_FLIP_PROB = 0.5
TRAIN_ROTATION_DEGREES = 15
TRAIN_BRIGHTNESS = 0.2
TRAIN_CONTRAST = 0.2

# Evaluation
NUM_WORKERS = 0  # Set to 0 for Windows compatibility
PIN_MEMORY = False