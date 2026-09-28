#!/usr/bin/env python3
"""
Final Training Readiness Check
Performs comprehensive verification before model training
"""

import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set

# Configuration
BASE_DIR = Path(r"C:\Users\Gayathri B\Documents\task4")
CLEAN_DATASET_DIR = BASE_DIR / "clean_dataset"
SPLITS_DIR = CLEAN_DATASET_DIR / "splits"
CLEAN_LABELS_PATH = CLEAN_DATASET_DIR / "clean_labels.csv"
CLEAN_IMAGES_DIR = CLEAN_DATASET_DIR / "images"

def load_csv(file_path: Path) -> List[Dict]:
    """Load CSV file into list of dictionaries"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def analyze_split_data() -> Dict:
    """Analyze train, validation, test splits"""
    print("Analyzing split data...")
    
    train_data = load_csv(SPLITS_DIR / "train.csv")
    val_data = load_csv(SPLITS_DIR / "validation.csv")
    test_data = load_csv(SPLITS_DIR / "test.csv")
    
    def get_statistics(data: List[Dict]) -> Dict:
        crop_counts = defaultdict(int)
        stage_counts = defaultdict(int)
        
        for row in data:
            crop = row['crop']
            stage = row['growth_stage']
            if crop:
                crop_counts[crop] += 1
            if stage:
                stage_counts[stage] += 1
        
        return {
            'total_images': len(data),
            'crop_counts': dict(crop_counts),
            'stage_counts': dict(stage_counts)
        }
    
    train_stats = get_statistics(train_data)
    val_stats = get_statistics(val_data)
    test_stats = get_statistics(test_data)
    
    print(f"TRAIN: {train_stats['total_images']} images")
    print(f"VALIDATION: {val_stats['total_images']} images")
    print(f"TEST: {test_stats['total_images']} images")
    
    return {
        'train': train_stats,
        'validation': val_stats,
        'test': test_stats
    }

def analyze_excluded_images() -> Dict:
    """Analyze the 69 excluded images"""
    print("Analyzing excluded images...")
    
    clean_labels = load_csv(CLEAN_LABELS_PATH)
    
    # Find images with missing labels
    missing_crop = []
    missing_stage = []
    missing_both = []
    
    for row in clean_labels:
        has_crop = bool(row['crop'].strip())
        has_stage = bool(row['growth_stage'].strip())
        
        if not has_crop and not has_stage:
            missing_both.append(row['image_file'])
        elif not has_crop:
            missing_crop.append(row['image_file'])
        elif not has_stage:
            missing_stage.append(row['image_file'])
    
    print(f"Missing crop: {len(missing_crop)}")
    print(f"Missing stage: {len(missing_stage)}")
    print(f"Missing both: {len(missing_both)}")
    
    return {
        'missing_crop_count': len(missing_crop),
        'missing_stage_count': len(missing_stage),
        'missing_both_count': len(missing_both),
        'missing_crop_files': missing_crop,
        'missing_stage_files': missing_stage,
        'missing_both_files': missing_both
    }

def analyze_small_class_warnings() -> List[Dict]:
    """Analyze the 12 small-class warnings"""
    print("Analyzing small-class warnings...")
    
    warnings = load_csv(SPLITS_DIR / "split_warnings.csv")
    
    # Get split data to count actual distribution
    train_data = load_csv(SPLITS_DIR / "train.csv")
    val_data = load_csv(SPLITS_DIR / "validation.csv")
    test_data = load_csv(SPLITS_DIR / "test.csv")
    
    def count_by_class(data: List[Dict], class_type: str) -> Dict:
        counts = defaultdict(int)
        for row in data:
            if class_type == 'crop':
                counts[row['crop']] += 1
            elif class_type == 'stage':
                counts[row['growth_stage']] += 1
        return dict(counts)
    
    train_crops = count_by_class(train_data, 'crop')
    train_stages = count_by_class(train_data, 'stage')
    val_crops = count_by_class(val_data, 'crop')
    val_stages = count_by_class(val_data, 'stage')
    test_crops = count_by_class(test_data, 'crop')
    test_stages = count_by_class(test_data, 'stage')
    
    detailed_warnings = []
    for warning in warnings:
        class_name = warning['class_name']
        class_type = warning['class_type']
        
        if class_type == 'crop':
            train_count = train_crops.get(class_name, 0)
            val_count = val_crops.get(class_name, 0)
            test_count = test_crops.get(class_name, 0)
        elif class_type == 'stage':
            train_count = train_stages.get(class_name, 0)
            val_count = val_stages.get(class_name, 0)
            test_count = test_stages.get(class_name, 0)
        else:  # crop_stage
            # Parse crop×stage combination
            parts = class_name.split(' × ')
            if len(parts) == 2:
                crop_name, stage_name = parts
                # Count images with this combination
                train_count = sum(1 for row in train_data if row['crop'] == crop_name and row['growth_stage'] == stage_name)
                val_count = sum(1 for row in val_data if row['crop'] == crop_name and row['growth_stage'] == stage_name)
                test_count = sum(1 for row in test_data if row['crop'] == crop_name and row['growth_stage'] == stage_name)
            else:
                train_count = val_count = test_count = 0
        
        detailed_warnings.append({
            'class_name': class_name,
            'class_type': class_type,
            'train_count': train_count,
            'validation_count': val_count,
            'test_count': test_count,
            'total_count': train_count + val_count + test_count,
            'original_count': int(warning['count']),
            'issue': warning['issue'],
            'recommended_action': warning['recommended_action']
        })
    
    print(f"Analyzed {len(detailed_warnings)} small-class warnings")
    
    return detailed_warnings

def verify_leakage_and_files() -> Dict:
    """Verify no leakage and all files exist"""
    print("Verifying leakage and file existence...")
    
    train_data = load_csv(SPLITS_DIR / "train.csv")
    val_data = load_csv(SPLITS_DIR / "validation.csv")
    test_data = load_csv(SPLITS_DIR / "test.csv")
    
    # Extract filenames and SHA-256
    train_files = set(Path(row['image_file']).name for row in train_data)
    val_files = set(Path(row['image_file']).name for row in val_data)
    test_files = set(Path(row['image_file']).name for row in test_data)
    
    train_hashes = set(row['sha256'] for row in train_data)
    val_hashes = set(row['sha256'] for row in val_data)
    test_hashes = set(row['sha256'] for row in test_data)
    
    # Check overlaps
    file_train_val = train_files & val_files
    file_train_test = train_files & test_files
    file_val_test = val_files & test_files
    
    hash_train_val = train_hashes & val_hashes
    hash_train_test = train_hashes & test_hashes
    hash_val_test = val_hashes & test_hashes
    
    # Check file existence
    all_files = train_files | val_files | test_files
    missing_files = []
    for filename in all_files:
        file_path = CLEAN_IMAGES_DIR / filename
        if not file_path.exists():
            missing_files.append(filename)
    
    # Check for missing labels in training data
    missing_labels_train = []
    for row in train_data:
        if not row['crop'].strip() or not row['growth_stage'].strip():
            missing_labels_train.append(row['image_file'])
    
    print(f"File overlaps: Train-Val={len(file_train_val)}, Train-Test={len(file_train_test)}, Val-Test={len(file_val_test)}")
    print(f"SHA-256 overlaps: Train-Val={len(hash_train_val)}, Train-Test={len(hash_train_test)}, Val-Test={len(hash_val_test)}")
    print(f"Missing files: {len(missing_files)}")
    print(f"Training images with missing labels: {len(missing_labels_train)}")
    
    return {
        'file_train_val_overlap': len(file_train_val),
        'file_train_test_overlap': len(file_train_test),
        'file_val_test_overlap': len(file_val_test),
        'hash_train_val_overlap': len(hash_train_val),
        'hash_train_test_overlap': len(hash_train_test),
        'hash_val_test_overlap': len(hash_val_test),
        'missing_files_count': len(missing_files),
        'missing_files': missing_files,
        'training_missing_labels_count': len(missing_labels_train),
        'training_missing_labels': missing_labels_train,
        'file_leakage_pass': len(file_train_val) == 0 and len(file_train_test) == 0 and len(file_val_test) == 0,
        'hash_leakage_pass': len(hash_train_val) == 0 and len(hash_train_test) == 0 and len(hash_val_test) == 0,
        'all_files_exist': len(missing_files) == 0,
        'all_training_have_labels': len(missing_labels_train) == 0
    }

def create_readiness_report(
    split_stats: Dict,
    excluded_analysis: Dict,
    small_class_warnings: List[Dict],
    verification: Dict
) -> List[Dict]:
    """Create final training readiness report"""
    print("Creating training readiness report...")
    
    report = []
    
    # Split statistics
    report.append({'category': 'SPLIT_STATISTICS', 'metric': 'train_total', 'value': split_stats['train']['total_images']})
    report.append({'category': 'SPLIT_STATISTICS', 'metric': 'validation_total', 'value': split_stats['validation']['total_images']})
    report.append({'category': 'SPLIT_STATISTICS', 'metric': 'test_total', 'value': split_stats['test']['total_images']})
    
    # Train crop distribution
    for crop, count in sorted(split_stats['train']['crop_counts'].items()):
        report.append({'category': 'TRAIN_CROP_DISTRIBUTION', 'metric': crop, 'value': count})
    
    # Train stage distribution
    for stage, count in sorted(split_stats['train']['stage_counts'].items()):
        report.append({'category': 'TRAIN_STAGE_DISTRIBUTION', 'metric': stage, 'value': count})
    
    # Validation crop distribution
    for crop, count in sorted(split_stats['validation']['crop_counts'].items()):
        report.append({'category': 'VALIDATION_CROP_DISTRIBUTION', 'metric': crop, 'value': count})
    
    # Validation stage distribution
    for stage, count in sorted(split_stats['validation']['stage_counts'].items()):
        report.append({'category': 'VALIDATION_STAGE_DISTRIBUTION', 'metric': stage, 'value': count})
    
    # Test crop distribution
    for crop, count in sorted(split_stats['test']['crop_counts'].items()):
        report.append({'category': 'TEST_CROP_DISTRIBUTION', 'metric': crop, 'value': count})
    
    # Test stage distribution
    for stage, count in sorted(split_stats['test']['stage_counts'].items()):
        report.append({'category': 'TEST_STAGE_DISTRIBUTION', 'metric': stage, 'value': count})
    
    # Excluded images
    report.append({'category': 'EXCLUDED_IMAGES', 'metric': 'missing_crop_count', 'value': excluded_analysis['missing_crop_count']})
    report.append({'category': 'EXCLUDED_IMAGES', 'metric': 'missing_stage_count', 'value': excluded_analysis['missing_stage_count']})
    report.append({'category': 'EXCLUDED_IMAGES', 'metric': 'missing_both_count', 'value': excluded_analysis['missing_both_count']})
    
    # Small class warnings
    for i, warning in enumerate(small_class_warnings, 1):
        report.append({'category': 'SMALL_CLASS_WARNING', 'metric': f'warning_{i}', 'value': f"{warning['class_name']} ({warning['class_type']})"})
        report.append({'category': 'SMALL_CLASS_WARNING', 'metric': f'warning_{i}_train', 'value': warning['train_count']})
        report.append({'category': 'SMALL_CLASS_WARNING', 'metric': f'warning_{i}_validation', 'value': warning['validation_count']})
        report.append({'category': 'SMALL_CLASS_WARNING', 'metric': f'warning_{i}_test', 'value': warning['test_count']})
    
    # Verification results
    report.append({'category': 'VERIFICATION', 'metric': 'file_leakage_pass', 'value': verification['file_leakage_pass']})
    report.append({'category': 'VERIFICATION', 'metric': 'hash_leakage_pass', 'value': verification['hash_leakage_pass']})
    report.append({'category': 'VERIFICATION', 'metric': 'all_files_exist', 'value': verification['all_files_exist']})
    report.append({'category': 'VERIFICATION', 'metric': 'all_training_have_labels', 'value': verification['all_training_have_labels']})
    
    print(f"Created report with {len(report)} entries")
    
    return report

def write_csv(data: List[Dict], output_path: Path) -> None:
    """Write data to CSV file"""
    if not data:
        print(f"Warning: No data to write to {output_path}")
        return
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Written {len(data)} rows to {output_path}")

def main():
    """Main execution function"""
    print("=" * 80)
    print("FINAL TRAINING READINESS CHECK")
    print("=" * 80)
    print()
    
    # Step 1: Analyze split data
    print("STEP 1: Analyzing split data...")
    split_stats = analyze_split_data()
    
    # Step 2: Analyze excluded images
    print("\nSTEP 2: Analyzing excluded images...")
    excluded_analysis = analyze_excluded_images()
    
    # Step 3: Analyze small-class warnings
    print("\nSTEP 3: Analyzing small-class warnings...")
    small_class_warnings = analyze_small_class_warnings()
    
    # Step 4: Verify leakage and files
    print("\nSTEP 4: Verifying leakage and file existence...")
    verification = verify_leakage_and_files()
    
    # Step 5: Create readiness report
    print("\nSTEP 5: Creating final training readiness report...")
    readiness_report = create_readiness_report(
        split_stats, excluded_analysis, small_class_warnings, verification
    )
    
    # Step 6: Write report
    print("\nSTEP 6: Writing training readiness report...")
    write_csv(readiness_report, SPLITS_DIR / "final_training_readiness.csv")
    
    # Also write detailed excluded images report
    excluded_report = []
    for filename in excluded_analysis['missing_both_files']:
        excluded_report.append({'filename': filename, 'missing': 'both'})
    for filename in excluded_analysis['missing_crop_files']:
        excluded_report.append({'filename': filename, 'missing': 'crop'})
    for filename in excluded_analysis['missing_stage_files']:
        excluded_report.append({'filename': filename, 'missing': 'stage'})
    
    write_csv(excluded_report, SPLITS_DIR / "excluded_images_detail.csv")
    
    # Print final summary
    print("\n" + "=" * 80)
    print("TRAINING READINESS SUMMARY")
    print("=" * 80)
    
    print("\nSPLIT STATISTICS:")
    print(f"  TRAIN: {split_stats['train']['total_images']} images")
    print(f"  VALIDATION: {split_stats['validation']['total_images']} images")
    print(f"  TEST: {split_stats['test']['total_images']} images")
    
    print("\nTRAIN CROP DISTRIBUTION:")
    for crop, count in sorted(split_stats['train']['crop_counts'].items()):
        print(f"  {crop}: {count}")
    
    print("\nTRAIN GROWTH-STAGE DISTRIBUTION:")
    for stage, count in sorted(split_stats['train']['stage_counts'].items()):
        print(f"  {stage}: {count}")
    
    print("\nEXCLUDED IMAGES (69 total):")
    print(f"  Missing crop: {excluded_analysis['missing_crop_count']}")
    print(f"  Missing growth-stage: {excluded_analysis['missing_stage_count']}")
    print(f"  Missing both: {excluded_analysis['missing_both_count']}")
    
    print("\nSMALL CLASS WARNINGS (12 total):")
    for warning in small_class_warnings:
        print(f"  {warning['class_name']} ({warning['class_type']}): Train={warning['train_count']}, Val={warning['validation_count']}, Test={warning['test_count']}")
    
    print("\nVERIFICATION:")
    print(f"  File leakage: {'PASS' if verification['file_leakage_pass'] else 'FAIL'}")
    print(f"  SHA-256 leakage: {'PASS' if verification['hash_leakage_pass'] else 'FAIL'}")
    print(f"  All files exist: {'PASS' if verification['all_files_exist'] else 'FAIL'}")
    print(f"  All training have labels: {'PASS' if verification['all_training_have_labels'] else 'FAIL'}")
    
    print("\n" + "=" * 80)
    print("TRAINING READINESS: READY" if all([
        verification['file_leakage_pass'],
        verification['hash_leakage_pass'],
        verification['all_files_exist'],
        verification['all_training_have_labels']
    ]) else "TRAINING READINESS: NOT READY")
    print("=" * 80)
    print("\nDo NOT train a model.")

if __name__ == "__main__":
    main()