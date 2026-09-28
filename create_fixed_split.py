"""
Create a fixed, reproducible train/validation/test split for all model experiments
Based on the same logic as the Colab notebook with seed 42
Uses only CSV operations (no pandas)
"""

import csv
import random
from pathlib import Path
import json

# Configuration
BASE_DIR = Path(r"C:\Users\Gayathri B\Documents\task4")
CLEAN_DATASET_DIR = BASE_DIR / "clean_dataset"
LABELS_CSV = CLEAN_DATASET_DIR / "clean_labels.csv"
SPLITS_DIR = CLEAN_DATASET_DIR / "splits"

# Set random seeds for reproducibility (same as Colab notebook)
SEED = 42
random.seed(SEED)

print("=" * 80)
print("CREATING FIXED TRAIN/VALIDATION/TEST SPLIT")
print("=" * 80)
print()

# Create splits directory
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# Load clean labels
print(f"Loading {LABELS_CSV.name}...")
labels = []
with open(LABELS_CSV, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    header = reader.fieldnames
    for row in reader:
        labels.append(row)

print(f"Total rows: {len(labels)}")
print(f"Columns: {header}")

# ------------------------------------------------------------
# 1. Remove rows where both crop and growth stage are missing
# ------------------------------------------------------------

complete = []
excluded = []

for row in labels:
    crop = row.get('crop', '').strip()
    stage = row.get('growth_stage', '').strip()
    
    if crop and stage:
        complete.append(row)
    else:
        excluded.append(row)

print(f"\nComplete labeled images: {len(complete)}")
print(f"Excluded images with both labels missing: {len(excluded)}")

# ------------------------------------------------------------
# 2. Stratification by crop (simplified approach)
# ------------------------------------------------------------

# Group by crop
crop_groups = {}
for row in complete:
    crop = row['crop']
    if crop not in crop_groups:
        crop_groups[crop] = []
    crop_groups[crop].append(row)

# Shuffle each crop group
for crop in crop_groups:
    random.shuffle(crop_groups[crop])

# Calculate split sizes per crop
train_data = []
val_data = []
test_data = []

for crop, crop_rows in crop_groups.items():
    n = len(crop_rows)
    train_n = int(0.70 * n)
    temp_n = n - train_n
    val_n = int(0.50 * temp_n)
    test_n = temp_n - val_n
    
    train_data.extend(crop_rows[:train_n])
    val_data.extend(crop_rows[train_n:train_n + val_n])
    test_data.extend(crop_rows[train_n + val_n:])

# Shuffle to mix crops
random.shuffle(train_data)
random.shuffle(val_data)
random.shuffle(test_data)

# ------------------------------------------------------------
# 3. Save split CSV files
# ------------------------------------------------------------

train_path = SPLITS_DIR / "train.csv"
val_path = SPLITS_DIR / "validation.csv"
test_path = SPLITS_DIR / "test.csv"

def write_csv(data, path, header):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(data)

write_csv(train_data, train_path, header)
write_csv(val_data, val_path, header)
write_csv(test_data, test_path, header)

# ------------------------------------------------------------
# 4. Print results
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("SPLIT CREATION COMPLETE")
print("=" * 80)

print(f"\nTRAIN      : {len(train_data)}")
print(f"VALIDATION : {len(val_data)}")
print(f"TEST       : {len(test_data)}")
print(f"EXCLUDED   : {len(excluded)}")

print("\nFiles created:")
print(f"  {train_path}")
print(f"  {val_path}")
print(f"  {test_path}")

# ------------------------------------------------------------
# 5. Show crop distributions
# ------------------------------------------------------------

def count_distribution(data, field_name):
    counts = {}
    for row in data:
        val = row[field_name]
        counts[val] = counts.get(val, 0) + 1
    return counts

print("\nTRAIN crop distribution:")
train_crop_dist = count_distribution(train_data, 'crop')
for crop in sorted(train_crop_dist.keys()):
    print(f"  {crop}: {train_crop_dist[crop]}")

print("\nVALIDATION crop distribution:")
val_crop_dist = count_distribution(val_data, 'crop')
for crop in sorted(val_crop_dist.keys()):
    print(f"  {crop}: {val_crop_dist[crop]}")

print("\nTEST crop distribution:")
test_crop_dist = count_distribution(test_data, 'crop')
for crop in sorted(test_crop_dist.keys()):
    print(f"  {crop}: {test_crop_dist[crop]}")

# ------------------------------------------------------------
# 6. Show stage distributions
# ------------------------------------------------------------

print("\nTRAIN growth-stage distribution:")
train_stage_dist = count_distribution(train_data, 'growth_stage')
for stage in sorted(train_stage_dist.keys()):
    print(f"  {stage}: {train_stage_dist[stage]}")

print("\nVALIDATION growth-stage distribution:")
val_stage_dist = count_distribution(val_data, 'growth_stage')
for stage in sorted(val_stage_dist.keys()):
    print(f"  {stage}: {val_stage_dist[stage]}")

print("\nTEST growth-stage distribution:")
test_stage_dist = count_distribution(test_data, 'growth_stage')
for stage in sorted(test_stage_dist.keys()):
    print(f"  {stage}: {test_stage_dist[stage]}")

# ------------------------------------------------------------
# 7. Save split summary
# ------------------------------------------------------------

summary = {
    'random_seed': SEED,
    'train_images': len(train_data),
    'validation_images': len(val_data),
    'test_images': len(test_data),
    'excluded_images': len(excluded),
    'total_images': len(labels),
    'train_percentage': f"{len(train_data)/len(complete)*100:.2f}%",
    'validation_percentage': f"{len(val_data)/len(complete)*100:.2f}%",
    'test_percentage': f"{len(test_data)/len(complete)*100:.2f}%",
    'crop_classes': sorted(train_crop_dist.keys()),
    'stage_classes': sorted(train_stage_dist.keys()),
    'crop_distribution': {
        'train': train_crop_dist,
        'validation': val_crop_dist,
        'test': test_crop_dist
    },
    'stage_distribution': {
        'train': train_stage_dist,
        'validation': val_stage_dist,
        'test': test_stage_dist
    }
}

with open(SPLITS_DIR / "split_config.json", 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 80)
print("FIXED SPLIT ESTABLISHED")
print("=" * 80)
print("\nThis split will be used for all model experiments:")
print("- ResNet-18")
print("- EfficientNet-B0") 
print("- MobileNet")
print("- Any other models")
print("\nAll experiments will use the same train/validation/test sets")
print("for fair performance comparison.")
print("=" * 80)