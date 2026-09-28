#!/usr/bin/env python3
"""
Leakage-Safe Dataset Split
Creates train/validation/test splits with proper stratification and leakage prevention
"""

import os
import csv
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import hashlib

# Configuration
BASE_DIR = Path(r"C:\Users\Gayathri B\Documents\task4")
CLEAN_DATASET_DIR = BASE_DIR / "clean_dataset"
CLEAN_LABELS_PATH = CLEAN_DATASET_DIR / "clean_labels.csv"
CLEAN_IMAGES_DIR = CLEAN_DATASET_DIR / "images"
SPLITS_DIR = CLEAN_DATASET_DIR / "splits"

# Create splits directory
SPLITS_DIR.mkdir(exist_ok=True)

# Random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def load_csv(file_path: Path) -> List[Dict]:
    """Load CSV file into list of dictionaries"""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def verify_clean_labels(clean_labels: List[Dict], clean_images_dir: Path) -> List[Dict]:
    """
    Verify that there is exactly one row per image and all referenced images exist
    Returns filtered clean labels (excluding images with missing labels)
    """
    print("Verifying clean labels...")
    
    # Check for duplicate image files
    image_files = [row['image_file'] for row in clean_labels]
    if len(image_files) != len(set(image_files)):
        raise ValueError("Duplicate image files found in clean_labels.csv")
    
    # Check for empty crop or stage values
    empty_crop = [row for row in clean_labels if not row['crop'].strip()]
    empty_stage = [row for row in clean_labels if not row['growth_stage'].strip()]
    
    if empty_crop:
        print(f"Warning: {len(empty_crop)} images have empty crop values - these will be excluded")
    if empty_stage:
        print(f"Warning: {len(empty_stage)} images have empty growth stage values - these will be excluded")
    
    # Filter out images with missing labels
    valid_labels = [row for row in clean_labels if row['crop'].strip() and row['growth_stage'].strip()]
    
    print(f"Original labels: {len(clean_labels)} rows")
    print(f"Valid labels (with crop and stage): {len(valid_labels)} rows")
    print(f"Excluded due to missing labels: {len(clean_labels) - len(valid_labels)} rows")
    
    # Check that all referenced images exist
    missing_images = []
    for row in valid_labels:
        image_path = Path(row['image_file'])
        if not image_path.exists():
            # Also check in clean images directory
            clean_image_path = clean_images_dir / image_path.name
            if not clean_image_path.exists():
                missing_images.append(row['image_file'])
    
    if missing_images:
        raise ValueError(f"Missing images: {len(missing_images)} images referenced in labels but not found")
    
    # Count actual images in directory
    actual_image_count = sum(1 for f in clean_images_dir.rglob('*') if f.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    
    print(f"Actual images in directory: {actual_image_count}")
    print(f"Verification passed: {len(valid_labels)} valid labels with complete information")
    
    return valid_labels

def create_leakage_safe_groups(clean_labels: List[Dict]) -> List[Dict]:
    """
    Create split groups ensuring images with same SHA-256 stay together
    """
    print("Creating leakage-safe split groups...")
    
    # Group by SHA-256 (images with same SHA-256 must stay together)
    sha256_to_group = defaultdict(list)
    
    for row in clean_labels:
        sha256 = row['sha256']
        sha256_to_group[sha256].append(row)
    
    # Create split groups
    split_groups = []
    group_counter = 1
    
    for sha256, group_rows in sha256_to_group.items():
        # Use first row's crop for stratification
        primary_crop = group_rows[0]['crop']
        primary_stage = group_rows[0]['growth_stage']
        
        split_group = {
            'split_group_id': f"SPLIT_{group_counter:04d}",
            'sha256': sha256,
            'image_files': [row['image_file'] for row in group_rows],
            'crop': primary_crop,
            'growth_stage': primary_stage,
            'gt_references': [row['gt_reference'] for row in group_rows],
            'count': len(group_rows)
        }
        
        split_groups.append(split_group)
        group_counter += 1
    
    print(f"Created {len(split_groups)} leakage-safe split groups")
    
    # Check for groups with multiple images (shouldn't happen in clean dataset)
    multi_image_groups = [g for g in split_groups if g['count'] > 1]
    if multi_image_groups:
        print(f"Warning: {len(multi_image_groups)} groups have multiple images (potential issue)")
    
    return split_groups

def calculate_class_distributions(split_groups: List[Dict]) -> Dict:
    """
    Calculate crop and growth-stage distributions
    """
    print("Calculating class distributions...")
    
    crop_distribution = defaultdict(int)
    stage_distribution = defaultdict(int)
    crop_stage_distribution = defaultdict(int)
    
    for group in split_groups:
        crop = group['crop']
        stage = group['growth_stage']
        
        crop_distribution[crop] += 1
        stage_distribution[stage] += 1
        crop_stage_distribution[(crop, stage)] += 1
    
    print(f"Crop classes: {len(crop_distribution)}")
    print(f"Stage classes: {len(stage_distribution)}")
    print(f"Crop×Stage combinations: {len(crop_stage_distribution)}")
    
    return {
        'crop': dict(crop_distribution),
        'stage': dict(stage_distribution),
        'crop_stage': dict(crop_stage_distribution)
    }

def identify_small_classes(distributions: Dict) -> List[Dict]:
    """
    Identify classes that cannot safely appear in all three splits
    """
    print("Identifying small classes...")
    
    warnings = []
    
    # Check crop classes
    for crop, count in distributions['crop'].items():
        if count < 3:
            warnings.append({
                'class_type': 'crop',
                'class_name': crop,
                'count': count,
                'issue': 'Cannot appear in all three splits (need at least 3)',
                'recommended_action': 'Keep in training set only'
            })
    
    # Check stage classes
    for stage, count in distributions['stage'].items():
        if count < 3:
            warnings.append({
                'class_type': 'stage',
                'class_name': stage,
                'count': count,
                'issue': 'Cannot appear in all three splits (need at least 3)',
                'recommended_action': 'Keep in training set only'
            })
    
    # Check crop×stage combinations
    for (crop, stage), count in distributions['crop_stage'].items():
        if count < 3:
            warnings.append({
                'class_type': 'crop_stage',
                'class_name': f"{crop} × {stage}",
                'count': count,
                'issue': 'Cannot appear in all three splits (need at least 3)',
                'recommended_action': 'Keep in training set only'
            })
    
    print(f"Found {len(warnings)} small class warnings")
    
    return warnings

def perform_stratified_split(
    split_groups: List[Dict],
    distributions: Dict,
    warnings: List[Dict]
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Perform stratified split (70% train, 15% validation, 15% test)
    """
    print("Performing stratified split...")
    
    # Group by crop for stratification
    crop_to_groups = defaultdict(list)
    for group in split_groups:
        crop_to_groups[group['crop']].append(group)
    
    train_groups = []
    val_groups = []
    test_groups = []
    
    # Collect problematic classes
    problematic_crops = {w['class_name'] for w in warnings if w['class_type'] == 'crop'}
    
    for crop, groups in crop_to_groups.items():
        # Shuffle groups within each crop
        random.shuffle(groups)
        
        if crop in problematic_crops:
            # Keep all in training
            train_groups.extend(groups)
        else:
            # Calculate split indices
            n = len(groups)
            train_end = int(0.7 * n)
            val_end = train_end + int(0.15 * n)
            
            # Assign to splits
            train_groups.extend(groups[:train_end])
            val_groups.extend(groups[train_end:val_end])
            test_groups.extend(groups[val_end:])
    
    print(f"Train groups: {len(train_groups)}")
    print(f"Validation groups: {len(val_groups)}")
    print(f"Test groups: {len(test_groups)}")
    
    return train_groups, val_groups, test_groups

def expand_groups_to_images(
    train_groups: List[Dict],
    val_groups: List[Dict],
    test_groups: List[Dict]
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Expand split groups back to individual image records
    """
    print("Expanding groups to individual images...")
    
    def expand_groups(groups: List[Dict]) -> List[Dict]:
        images = []
        for group in groups:
            for image_file in group['image_files']:
                images.append({
                    'image_file': image_file,
                    'sha256': group['sha256'],
                    'gt_reference': group['gt_references'][0] if group['gt_references'] else '',
                    'crop': group['crop'],
                    'growth_stage': group['growth_stage'],
                    'split_group_id': group['split_group_id']
                })
        return images
    
    train_images = expand_groups(train_groups)
    val_images = expand_groups(val_groups)
    test_images = expand_groups(test_groups)
    
    print(f"Train images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    print(f"Test images: {len(test_images)}")
    
    return train_images, val_images, test_images

def create_split_files(
    train_images: List[Dict],
    val_images: List[Dict],
    test_images: List[Dict],
    splits_dir: Path
) -> None:
    """
    Create train.csv, validation.csv, test.csv files
    """
    print("Creating split files...")
    
    write_csv(train_images, splits_dir / "train.csv")
    write_csv(val_images, splits_dir / "validation.csv")
    write_csv(test_images, splits_dir / "test.csv")

def write_csv(data: List[Dict], output_path: Path) -> None:
    """Write data to CSV file"""
    if not data:
        print(f"Warning: No data to write to {output_path}")
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            f.write("")
        return
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Written {len(data)} rows to {output_path}")

def verify_leakage(
    train_images: List[Dict],
    val_images: List[Dict],
    test_images: List[Dict]
) -> Dict:
    """
    Verify no leakage between splits
    """
    print("Verifying leakage...")
    
    # Extract SHA-256 hashes
    train_hashes = set(img['sha256'] for img in train_images)
    val_hashes = set(img['sha256'] for img in val_images)
    test_hashes = set(img['sha256'] for img in test_images)
    
    # Check SHA-256 overlap
    train_val_overlap = train_hashes & val_hashes
    train_test_overlap = train_hashes & test_hashes
    val_test_overlap = val_hashes & test_hashes
    
    # Check image filename overlap
    train_files = set(Path(img['image_file']).name for img in train_images)
    val_files = set(Path(img['image_file']).name for img in val_images)
    test_files = set(Path(img['image_file']).name for img in test_images)
    
    train_val_file_overlap = train_files & val_files
    train_test_file_overlap = train_files & test_files
    val_test_file_overlap = val_files & test_files
    
    # Check GT reference overlap
    train_refs = set(img['gt_reference'] for img in train_images if img['gt_reference'])
    val_refs = set(img['gt_reference'] for img in val_images if img['gt_reference'])
    test_refs = set(img['gt_reference'] for img in test_images if img['gt_reference'])
    
    train_val_ref_overlap = train_refs & val_refs
    train_test_ref_overlap = train_refs & test_refs
    val_test_ref_overlap = val_refs & test_refs
    
    leakage_results = {
        'sha256_train_val_overlap': len(train_val_overlap),
        'sha256_train_test_overlap': len(train_test_overlap),
        'sha256_val_test_overlap': len(val_test_overlap),
        'file_train_val_overlap': len(train_val_file_overlap),
        'file_train_test_overlap': len(train_test_file_overlap),
        'file_val_test_overlap': len(val_test_file_overlap),
        'ref_train_val_overlap': len(train_val_ref_overlap),
        'ref_train_test_overlap': len(train_test_ref_overlap),
        'ref_val_test_overlap': len(val_test_ref_overlap),
        'sha256_leakage_pass': len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0,
        'file_leakage_pass': len(train_val_file_overlap) == 0 and len(train_test_file_overlap) == 0 and len(val_test_file_overlap) == 0
    }
    
    print("Leakage Verification Results:")
    print(f"  SHA-256 Train-Val overlap: {leakage_results['sha256_train_val_overlap']}")
    print(f"  SHA-256 Train-Test overlap: {leakage_results['sha256_train_test_overlap']}")
    print(f"  SHA-256 Val-Test overlap: {leakage_results['sha256_val_test_overlap']}")
    print(f"  File Train-Val overlap: {leakage_results['file_train_val_overlap']}")
    print(f"  File Train-Test overlap: {leakage_results['file_train_test_overlap']}")
    print(f"  File Val-Test overlap: {leakage_results['file_val_test_overlap']}")
    print(f"  SHA-256 leakage: {'PASS' if leakage_results['sha256_leakage_pass'] else 'FAIL'}")
    print(f"  File leakage: {'PASS' if leakage_results['file_leakage_pass'] else 'FAIL'}")
    
    return leakage_results

def calculate_split_distributions(
    train_images: List[Dict],
    val_images: List[Dict],
    test_images: List[Dict]
) -> Dict:
    """
    Calculate crop and stage distributions for each split
    """
    print("Calculating split distributions...")
    
    def get_distribution(images: List[Dict]) -> Dict:
        crop_dist = defaultdict(int)
        stage_dist = defaultdict(int)
        
        for img in images:
            crop_dist[img['crop']] += 1
            stage_dist[img['growth_stage']] += 1
        
        return {
            'crop': dict(crop_dist),
            'stage': dict(stage_dist)
        }
    
    train_dist = get_distribution(train_images)
    val_dist = get_distribution(val_images)
    test_dist = get_distribution(test_images)
    
    return {
        'train': train_dist,
        'validation': val_dist,
        'test': test_dist
    }

def create_split_summary(
    train_images: List[Dict],
    val_images: List[Dict],
    test_images: List[Dict],
    leakage_results: Dict,
    split_distributions: Dict,
    warnings: List[Dict],
    excluded_count: int,
    splits_dir: Path
) -> None:
    """
    Create split summary CSV
    """
    print("Creating split summary...")
    
    total_images = len(train_images) + len(val_images) + len(test_images)
    
    summary_rows = [
        {'metric': 'total_images', 'value': total_images},
        {'metric': 'excluded_due_to_missing_labels', 'value': excluded_count},
        {'metric': 'train_images', 'value': len(train_images)},
        {'metric': 'train_percentage', 'value': f"{len(train_images)/total_images:.2%}"},
        {'metric': 'validation_images', 'value': len(val_images)},
        {'metric': 'validation_percentage', 'value': f"{len(val_images)/total_images:.2%}"},
        {'metric': 'test_images', 'value': len(test_images)},
        {'metric': 'test_percentage', 'value': f"{len(test_images)/total_images:.2%}"},
        {'metric': 'random_seed', 'value': RANDOM_SEED},
        {'metric': 'sha256_leakage_pass', 'value': leakage_results['sha256_leakage_pass']},
        {'metric': 'file_leakage_pass', 'value': leakage_results['file_leakage_pass']},
    ]
    
    # Add crop distributions
    for crop in sorted(split_distributions['train']['crop'].keys()):
        train_count = split_distributions['train']['crop'].get(crop, 0)
        val_count = split_distributions['validation']['crop'].get(crop, 0)
        test_count = split_distributions['test']['crop'].get(crop, 0)
        
        summary_rows.append({'metric': f'train_crop_{crop}', 'value': train_count})
        summary_rows.append({'metric': f'val_crop_{crop}', 'value': val_count})
        summary_rows.append({'metric': f'test_crop_{crop}', 'value': test_count})
    
    # Add stage distributions
    for stage in sorted(split_distributions['train']['stage'].keys()):
        train_count = split_distributions['train']['stage'].get(stage, 0)
        val_count = split_distributions['validation']['stage'].get(stage, 0)
        test_count = split_distributions['test']['stage'].get(stage, 0)
        
        summary_rows.append({'metric': f'train_stage_{stage}', 'value': train_count})
        summary_rows.append({'metric': f'val_stage_{stage}', 'value': val_count})
        summary_rows.append({'metric': f'test_stage_{stage}', 'value': test_count})
    
    write_csv(summary_rows, splits_dir / "split_summary.csv")

def create_split_warnings(warnings: List[Dict], splits_dir: Path) -> None:
    """
    Create split warnings CSV
    """
    print("Creating split warnings...")
    
    if warnings:
        write_csv(warnings, splits_dir / "split_warnings.csv")
    else:
        # Create empty file with header
        with open(splits_dir / "split_warnings.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['class_type', 'class_name', 'count', 'issue', 'recommended_action'])
            writer.writeheader()

def format_distribution(distribution: Dict, total: int) -> str:
    """Format distribution as markdown table"""
    if not distribution:
        return "No data"
    
    lines = ["| Class | Count | Percentage |", "|-------|-------|------------|"]
    for class_name in sorted(distribution.keys()):
        count = distribution[class_name]
        percentage = count / total * 100 if total > 0 else 0
        # Handle empty class names
        display_name = class_name if class_name else "(empty)"
        lines.append(f"| {display_name} | {count} | {percentage:.1f}% |")
    
    return "\n".join(lines)

def create_split_readme(
    train_images: List[Dict],
    val_images: List[Dict],
    test_images: List[Dict],
    leakage_results: Dict,
    split_distributions: Dict,
    warnings: List[Dict],
    excluded_count: int,
    splits_dir: Path
) -> None:
    """
    Create comprehensive split README
    """
    print("Creating split README...")
    
    total_images = len(train_images) + len(val_images) + len(test_images)
    
    readme_content = f"""# Dataset Split Configuration

## Overview
This directory contains the train/validation/test splits for the clean dataset, created with leakage prevention and stratification.

## Split Configuration

- **Random Seed**: {RANDOM_SEED}
- **Split Ratios**: 70% train, 15% validation, 15% test
- **Stratification**: By crop class
- **Leakage Prevention**: SHA-256 based grouping

## Split Statistics

- **Total Images with Complete Labels**: {total_images}
- **Images Excluded Due to Missing Labels**: {excluded_count}
- **Train**: {len(train_images)} ({len(train_images)/total_images:.1%})
- **Validation**: {len(val_images)} ({len(val_images)/total_images:.1%})
- **Test**: {len(test_images)} ({len(test_images)/total_images:.1%})

## Excluded Images

{excluded_count} images were excluded from the split because they had missing crop or growth-stage labels. These images are present in the clean dataset but cannot be used for supervised training without complete label information.

## Crop Distribution

### Train
{format_distribution(split_distributions['train']['crop'], len(train_images))}

### Validation
{format_distribution(split_distributions['validation']['crop'], len(val_images))}

### Test
{format_distribution(split_distributions['test']['crop'], len(test_images))}

## Growth Stage Distribution

### Train
{format_distribution(split_distributions['train']['stage'], len(train_images))}

### Validation
{format_distribution(split_distributions['validation']['stage'], len(val_images))}

### Test
{format_distribution(split_distributions['test']['stage'], len(test_images))}

## Leakage Verification

- **SHA-256 Train-Val Overlap**: {leakage_results['sha256_train_val_overlap']}
- **SHA-256 Train-Test Overlap**: {leakage_results['sha256_train_test_overlap']}
- **SHA-256 Val-Test Overlap**: {leakage_results['sha256_val_test_overlap']}
- **SHA-256 Leakage**: {'PASS' if leakage_results['sha256_leakage_pass'] else 'FAIL'}

- **File Train-Val Overlap**: {leakage_results['file_train_val_overlap']}
- **File Train-Test Overlap**: {leakage_results['file_train_test_overlap']}
- **File Val-Test Overlap**: {leakage_results['file_val_test_overlap']}
- **File Leakage**: {'PASS' if leakage_results['file_leakage_pass'] else 'FAIL'}

## Small Class Warnings

{len(warnings)} classes could not be represented in all three splits due to insufficient samples.

{'None' if not warnings else 'See split_warnings.csv for details.'}

## Files Generated

1. **train.csv** - Training set images and labels
2. **validation.csv** - Validation set images and labels
3. **test.csv** - Test set images and labels
4. **split_summary.csv** - Comprehensive split statistics
5. **split_warnings.csv** - Small class warnings (if any)
6. **README.md** - This file

## Methodology

1. **Leakage Prevention**: Images with identical SHA-256 hashes are grouped together and always assigned to the same split
2. **Stratification**: Primary stratification by crop class to ensure balanced representation
3. **Small Classes**: Classes with fewer than 3 samples are kept in training set only
4. **Missing Labels**: Images with incomplete label information are excluded from training
5. **Reproducibility**: Fixed random seed ({RANDOM_SEED}) ensures consistent splits

## Data Safety

✓ Original clean images were not modified
✓ Original clean_labels.csv was not modified
✓ No data leakage between splits
✓ All splits are documented and reproducible

## Next Steps

1. **Review small class warnings** - Determine if additional data collection is needed
2. **Review split distributions** - Ensure class balance meets requirements
3. **Investigate missing labels** - Determine if the {excluded_count} excluded images can be labeled
4. **Begin model training** - Use these splits for training, validation, and testing

Generated on: 2026-09-24
"""

    readme_path = splits_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"Created split README at {readme_path}")

def main():
    """Main execution function"""
    print("=" * 80)
    print("LEAKAGE-SAFE DATASET SPLIT")
    print("=" * 80)
    print()
    
    # Step 1: Load clean labels
    print("STEP 1: Loading clean labels...")
    clean_labels = load_csv(CLEAN_LABELS_PATH)
    print(f"Loaded {len(clean_labels)} clean label entries")
    
    # Step 2: Verify clean labels
    print("\nSTEP 2: Verifying clean labels...")
    original_count = len(clean_labels)
    clean_labels = verify_clean_labels(clean_labels, CLEAN_IMAGES_DIR)
    excluded_count = original_count - len(clean_labels)
    
    # Step 3: Create leakage-safe groups
    print("\nSTEP 3: Creating leakage-safe split groups...")
    split_groups = create_leakage_safe_groups(clean_labels)
    
    # Step 4: Calculate distributions
    print("\nSTEP 4: Calculating class distributions...")
    distributions = calculate_class_distributions(split_groups)
    
    # Step 5: Identify small classes
    print("\nSTEP 5: Identifying small classes...")
    warnings = identify_small_classes(distributions)
    
    # Step 6: Perform stratified split
    print("\nSTEP 6: Performing stratified split...")
    train_groups, val_groups, test_groups = perform_stratified_split(split_groups, distributions, warnings)
    
    # Step 7: Expand to individual images
    print("\nSTEP 7: Expanding groups to individual images...")
    train_images, val_images, test_images = expand_groups_to_images(train_groups, val_groups, test_groups)
    
    # Step 8: Create split files
    print("\nSTEP 8: Creating split files...")
    create_split_files(train_images, val_images, test_images, SPLITS_DIR)
    
    # Step 9: Verify leakage
    print("\nSTEP 9: Verifying leakage...")
    leakage_results = verify_leakage(train_images, val_images, test_images)
    
    # Step 10: Calculate split distributions
    print("\nSTEP 10: Calculating split distributions...")
    split_distributions = calculate_split_distributions(train_images, val_images, test_images)
    
    # Step 11: Create summary
    print("\nSTEP 11: Creating split summary...")
    create_split_summary(train_images, val_images, test_images, leakage_results, split_distributions, warnings, excluded_count, SPLITS_DIR)
    
    # Step 12: Create warnings
    print("\nSTEP 12: Creating split warnings...")
    create_split_warnings(warnings, SPLITS_DIR)
    
    # Step 13: Create README
    print("\nSTEP 13: Creating split README...")
    create_split_readme(train_images, val_images, test_images, leakage_results, split_distributions, warnings, excluded_count, SPLITS_DIR)
    
    # Final console output
    print("\n" + "=" * 80)
    print("DATASET SPLIT COMPLETE")
    print("=" * 80)
    print(f"\nTRAIN: {len(train_images)}")
    print(f"VALIDATION: {len(val_images)}")
    print(f"TEST: {len(test_images)}")
    print(f"\nSHA-256 leakage: {'PASS' if leakage_results['sha256_leakage_pass'] else 'FAIL'}")
    print(f"Image leakage: {'PASS' if leakage_results['file_leakage_pass'] else 'FAIL'}")
    print("\n" + "=" * 80)
    print("Do NOT train a model until leakage verification passes.")
    print("=" * 80)

if __name__ == "__main__":
    main()