# FarmwiseAI Task 4 - Agricultural Image Classification

Multi-task crop and growth-stage classification using deep learning models.

## Project Overview

This project focuses on training independent models for agricultural image classification without using external AI APIs (Bedrock, Gemini, etc.). The goal is to establish reliable baselines for crop and growth-stage prediction.

## Dataset

- **Total Images**: 3,262 clean images
- **Crop Classes**: 12 (Rice, Coconut, Sugarcane, Maize, Banana, etc.)
- **Growth-Stage Classes**: 5 (vegetation, full_growth, flowering, sown, harvesting)
- **Data Source**: Ground Truth Points CSV with image references

## Data Pipeline

1. **Dataset Audit**: Initial audit and image matching
2. **Final Mapping**: Image ↔ Ground Truth mapping with SHA-256 duplicate detection
3. **Validation**: Discrepancy analysis between audit and mapping
4. **Clean Dataset**: Removal of duplicates, conflicts, and missing labels
5. **Fixed Split**: Train/validation/test split (70/15/15) with seed 42

## Fixed Split

- **Train**: 2,230 images
- **Validation**: 478 images
- **Test**: 485 images
- **Random Seed**: 42 (reproducible)

All model experiments use this fixed split for fair comparison.

## Project Structure

```
task4/
├── clean_dataset/
│   ├── images/                    # Clean image files (not in git)
│   ├── clean_labels.csv          # Final labeled dataset
│   ├── split_candidates.csv      # Split preparation data
│   └── splits/
│       ├── train.csv             # Training split
│       ├── validation.csv        # Validation split
│       ├── test.csv              # Test split
│       └── split_config.json     # Split configuration
├── final_dataset_mapping/        # Image ↔ GT mapping results
├── mapping_validation/           # Validation audit results
├── model_training/               # Model training scripts
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── inference.py
├── model_training_colab.ipynb     # Colab training notebook
└── create_fixed_split.py         # Split generation script
```

## Model Experiments

### Models Tested
- ResNet-18 (baseline)
- EfficientNet-B0
- MobileNet
- Additional architectures for comparison

### Training Approach
- Multi-task learning (crop + growth-stage)
- Class-weighted loss for imbalanced data
- Data augmentation
- Early stopping
- Fixed random seed for reproducibility

## Usage

### Local Training
```bash
# Create fixed split (run once)
python create_fixed_split.py

# Train model
cd model_training
python train_model.py

# Evaluate model
python evaluate.py
```

### Colab Training
1. Upload `clean_dataset.zip` to Google Drive
2. Open `model_training_colab.ipynb` in Colab
3. Update paths to match your Drive structure
4. Run cells in order

## Key Results

### Dataset Summary
- Original images: 3,421
- Duplicates removed: 129
- Conflicts excluded: 10
- Final clean images: 3,262
- Usable for training: 3,193

### Performance Metrics
- Crop classification accuracy: ~90% (validation)
- Growth-stage classification accuracy: ~74% (validation)
- Joint crop+stage accuracy: ~68% (validation)

### Limitations
- Some crop classes have very few samples (≤2)
- Test set has zero samples for some rare classes
- Overall accuracy doesn't reflect per-class performance
- Need stratified k-fold cross-validation for better evaluation

## Important Notes

- **No external AI APIs**: All predictions from trained models
- **No original data modification**: All processing preserves original files
- **Fixed split**: All experiments use same train/validation/test sets
- **Reproducible**: Random seed 42 for consistent results
- **No model training yet**: Dataset preparation complete, ready for training

## Setup Instructions

1. Clone this repository
2. Place `clean_dataset.zip` in the project directory
3. Extract the zip file
4. Run `python create_fixed_split.py` to generate splits
5. Use model training scripts or Colab notebook

## Dependencies

- Python 3.8+
- PyTorch
- torchvision
- PIL/Pillow
- matplotlib
- seaborn

## License

Proprietary - FarmwiseAI