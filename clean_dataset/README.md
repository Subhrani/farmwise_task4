# Clean Training Dataset

## Overview
This directory contains a clean training dataset prepared based on the validation audit results. All duplicates, conflicts, and ambiguities have been resolved or excluded.

## Dataset Statistics

- **Original images**: 3421
- **Duplicate copies excluded**: 129
- **Conflicting duplicate images excluded**: 2
- **Stage conflict images excluded**: 8
- **Missing GT references**: 2
- **Final clean images**: 3262

## Data Quality Status

- **Crop classes**: 12
- **Growth-stage classes**: 5

## Files Generated

### 1. duplicate_removal_log.csv
Documents which duplicate images were kept and which were removed for each SHA-256 group.

### 2. stage_conflict_log.csv
Lists the 8 images with growth-stage label conflicts that were excluded from training.

### 3. missing_ground_truth_references.csv
Lists the 2 Ground Truth references with no corresponding image files.

### 4. clean_labels.csv
Primary training labels file - one row per clean image with all metadata.

### 5. split_candidates.csv
Prepared for dataset splitting - ensures exact duplicates stay in same split.

### 6. clean_dataset_summary.csv
Comprehensive statistics and class distributions.

### 7. images/
Directory containing only the clean, non-duplicate images ready for training.

## Excluded Items

### Duplicate Images
- 129 exact duplicate copies were removed
- One representative image kept per SHA-256 group
- Prevents data leakage in train/validation/test splits

### Conflicting Duplicate Group
- 2 images excluded due to label conflicts
- Group marked as REQUIRES_REVIEW
- Same image content (SHA-256) with conflicting Ground Truth labels

### Stage Conflicts
- 8 images excluded due to growth-stage conflicts
- Same image associated with multiple growth-stage labels
- Marked as REQUIRES_REVIEW

### Missing GT References
- 2 Ground Truth references have no corresponding images
- Cannot be used for image-based training

## Data Safety

✓ Original images were not modified
✓ Original Ground Truth CSV was not modified
✓ All exclusions are documented in log files
✓ No labels were invented or changed
✓ No conflicts were automatically resolved
✓ No model training was performed

## Next Steps

1. **Review conflicting duplicate group** - Determine correct labels for the excluded group
2. **Review stage conflicts** - Determine correct growth stages for the 8 excluded images
3. **Review missing GT references** - Investigate if images exist elsewhere
4. **Dataset splitting** - Use split_candidates.csv to create train/validation/test splits
5. **Model training** - Only proceed after all conflicts are resolved

## Important Notes

- All images in this dataset have unique SHA-256 hashes
- Each image has a one-to-one mapping to Ground Truth
- No label conflicts exist in the clean dataset
- Ready for safe model training without data leakage risks

Generated on: 2026-09-24
