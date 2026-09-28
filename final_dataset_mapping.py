#!/usr/bin/env python3
"""
Final Dataset Mapping Script
Creates comprehensive mapping between actual images and Ground Truth data
Based on corrected audit logic from dataset_audit_v2
"""

import os
import csv
import hashlib
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import re

# Configuration
BASE_DIR = Path(r"C:\Users\Gayathri B\Documents\task4")
IMAGES_DIR = BASE_DIR / "images"
GROUND_TRUTH_CSV = BASE_DIR / "Ground_Truth_Points" / "Ground_truth_Points.csv"
AUDIT_V2_DIR = BASE_DIR / "dataset_audit_v2"
OUTPUT_DIR = BASE_DIR / "final_dataset_mapping"

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def inventory_images(images_dir: Path) -> List[Dict]:
    """
    PART 1: Inventory all actual images
    Returns list of image metadata dictionaries
    """
    print("PART 1: Inventorying all actual images...")
    
    image_inventory = []
    image_extensions = {'.jpg', '.jpeg', '.png'}
    
    for image_file in images_dir.rglob('*'):
        if image_file.suffix.lower() in image_extensions:
            try:
                file_size = image_file.stat().st_size
                sha256 = calculate_sha256(image_file)
                relative_path = image_file.relative_to(images_dir)
                
                image_inventory.append({
                    'image_file': str(image_file),
                    'relative_image_path': str(relative_path),
                    'filename': image_file.name,
                    'file_extension': image_file.suffix.lower(),
                    'file_size_bytes': file_size,
                    'sha256': sha256
                })
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
    
    print(f"Found {len(image_inventory)} image files")
    return image_inventory

def parse_ground_truth(csv_path: Path) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    """
    PART 2: Parse Ground Truth CSV
    Returns (rows_dict, image_references_map)
    """
    print("PART 2: Parsing Ground Truth CSV...")
    
    gt_rows = []
    image_references_map = defaultdict(list)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row_num, row in enumerate(reader, 1):
            gt_rows.append(row)
            
            # Parse image_path column (semicolon-separated)
            image_path_raw = row.get('image_path', '')
            if image_path_raw:
                # Split by semicolon and clean up each reference
                references = [ref.strip() for ref in image_path_raw.split(';') if ref.strip()]
                
                for ref in references:
                    # Normalize reference (remove quotes, paths, extensions)
                    normalized_ref = ref.strip('"\'').replace('\\', '/').replace('/', '_')
                    # Remove common path prefixes
                    normalized_ref = re.sub(r'^.*?images/', '', normalized_ref, flags=re.IGNORECASE)
                    normalized_ref = re.sub(r'^.*?Images/', '', normalized_ref, flags=re.IGNORECASE)
                    # Remove extension if present
                    normalized_ref = re.sub(r'\.(jpg|jpeg|png)$', '', normalized_ref, flags=re.IGNORECASE)
                    
                    if normalized_ref:
                        image_references_map[normalized_ref].append({
                            'row': row,
                            'row_num': row_num,
                            'original_ref': ref
                        })
    
    print(f"Loaded {len(gt_rows)} ground truth rows")
    print(f"Found {len(image_references_map)} unique image references")
    
    return gt_rows, image_references_map

def load_corrected_matching(audit_v2_dir: Path) -> Dict[str, Dict]:
    """
    Load corrected matching results from dataset_audit_v2
    """
    print("Loading corrected matching results from dataset_audit_v2...")
    
    matching_report_path = audit_v2_dir / "corrected_image_matching_report.csv"
    matching_map = {}
    
    if matching_report_path.exists():
        with open(matching_report_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                gt_ref = row['ground_truth_image_reference']
                matching_map[gt_ref] = row
    
    print(f"Loaded {len(matching_map)} corrected matches")
    return matching_map

def match_images_to_ground_truth(
    image_inventory: List[Dict],
    image_references_map: Dict[str, List[Dict]],
    corrected_matching: Dict[str, Dict]
) -> Tuple[Dict[str, Dict], Dict[str, List[str]], Dict[str, str]]:
    """
    PART 3: Image ↔ Ground Truth Matching
    Returns (image_to_gt_map, gt_to_images_map, image_match_status)
    """
    print("PART 3: Performing Image ↔ Ground Truth Matching...")
    
    image_to_gt_map = defaultdict(list)  # image_file -> list of gt references
    gt_to_images_map = defaultdict(list)  # gt_reference -> list of image_files
    image_match_status = {}  # image_file -> MATCHED_UNIQUE, MATCHED_MULTIPLE, NOT_FOUND
    
    # Build filename lookup for quick access
    filename_to_images = defaultdict(list)
    for img in image_inventory:
        filename_to_images[img['filename']].append(img)
    
    # Process each ground truth reference
    for gt_ref, gt_data_list in image_references_map.items():
        # Check if we have corrected matching data
        if gt_ref in corrected_matching:
            corrected_match = corrected_matching[gt_ref]
            matched_file = corrected_match['matched_image_file']
            
            # Extract just the filename from the full path
            matched_filename = Path(matched_file).name
            
            # Find all images with this filename
            if matched_filename in filename_to_images:
                for img in filename_to_images[matched_filename]:
                    image_to_gt_map[img['image_file']].append({
                        'gt_reference': gt_ref,
                        'gt_data': gt_data_list
                    })
                    gt_to_images_map[gt_ref].append(img['image_file'])
        else:
            # Fallback to substring matching (as per original corrected logic)
            for img in image_inventory:
                filename = img['filename']
                # Check if gt_ref is a substring of filename
                if gt_ref.lower() in filename.lower():
                    image_to_gt_map[img['image_file']].append({
                        'gt_reference': gt_ref,
                        'gt_data': gt_data_list
                    })
                    gt_to_images_map[gt_ref].append(img['image_file'])
    
    # Determine match status for each image
    for img in image_inventory:
        matches = image_to_gt_map.get(img['image_file'], [])
        if len(matches) == 1:
            image_match_status[img['image_file']] = 'MATCHED_UNIQUE'
        elif len(matches) > 1:
            image_match_status[img['image_file']] = 'MATCHED_MULTIPLE'
        else:
            image_match_status[img['image_file']] = 'NOT_FOUND'
    
    # Count statistics
    unique_matches = sum(1 for status in image_match_status.values() if status == 'MATCHED_UNIQUE')
    multiple_matches = sum(1 for status in image_match_status.values() if status == 'MATCHED_MULTIPLE')
    not_found = sum(1 for status in image_match_status.values() if status == 'NOT_FOUND')
    
    print(f"Unique matches: {unique_matches}")
    print(f"Multiple matches: {multiple_matches}")
    print(f"Not found: {not_found}")
    
    return dict(image_to_gt_map), dict(gt_to_images_map), image_match_status

def create_duplicate_mapping(image_inventory: List[Dict]) -> Tuple[Dict[str, str], Dict[str, List[Dict]]]:
    """
    PART 4: SHA-256 Duplicate Mapping
    Returns (sha256_to_group, duplicate_groups)
    """
    print("PART 4: Creating SHA-256 Duplicate Mapping...")
    
    sha256_to_images = defaultdict(list)
    for img in image_inventory:
        sha256_to_images[img['sha256']].append(img)
    
    sha256_to_group = {}
    duplicate_groups = {}
    group_counter = 1
    
    for sha256, images in sha256_to_images.items():
        if len(images) > 1:
            group_id = f"DUP_{group_counter:04d}"
            group_counter += 1
            duplicate_groups[group_id] = images
            for img in images:
                sha256_to_group[sha256] = group_id
        else:
            sha256_to_group[sha256] = "UNIQUE"
    
    unique_hashes = len([sha256 for sha256, images in sha256_to_images.items() if len(images) == 1])
    duplicate_hash_count = len([sha256 for sha256, images in sha256_to_images.items() if len(images) > 1])
    duplicate_image_count = sum(len(images) for images in sha256_to_images.values() if len(images) > 1)
    
    print(f"Unique SHA-256 hashes: {unique_hashes}")
    print(f"Duplicate groups: {duplicate_hash_count}")
    print(f"Images in duplicate groups: {duplicate_image_count}")
    
    return sha256_to_group, duplicate_groups

def extract_labels_from_matches(matches: List[Dict]) -> Tuple[Set[str], Set[str], Dict]:
    """
    Extract crop and growth stage labels from matches
    Returns (crops, stages, metadata)
    """
    crops = set()
    stages = set()
    metadata = {}
    
    for match in matches:
        gt_data = match['gt_data']
        for gt_entry in gt_data:
            row = gt_entry['row']
            crop = row.get('final_crop_name', '').strip()
            stage = row.get('final_crop_stage', '').strip()
            
            if crop:
                crops.add(crop)
            if stage:
                stages.add(stage)
            
            # Store first occurrence of metadata
            if not metadata:
                metadata = {
                    'Village_LG': row.get('Village LG', ''),
                    'district': row.get('district_1', ''),
                    'taluk': row.get('taluk_co_1', ''),
                    'village': row.get('village__1', ''),
                    'survey_number': row.get('survey_n_1', ''),
                    'sub_division': row.get('sub_divi_1', ''),
                    'final_lat': row.get('final_lat', ''),
                    'final_long': row.get('final_long', ''),
                    'timestamp': row.get('timestamp', ''),
                    'image_timestamp': row.get('image_timestamp', ''),
                    'image_latitude': row.get('image_latitude', ''),
                    'image_longitude': row.get('image_longitude', '')
                }
    
    return crops, stages, metadata

def determine_mapping_confidence(
    match_status: str,
    crops: Set[str],
    stages: Set[str],
    candidate_count: int
) -> str:
    """
    Determine mapping confidence level
    """
    if match_status == 'NOT_FOUND':
        return 'UNRESOLVED'
    
    if match_status == 'MATCHED_UNIQUE':
        if len(crops) == 1 and len(stages) <= 1:
            return 'HIGH'
        elif len(crops) > 1 or len(stages) > 1:
            return 'LOW'  # Label conflict
        else:
            return 'MEDIUM'
    
    if match_status == 'MATCHED_MULTIPLE':
        if len(crops) == 1 and len(stages) <= 1:
            return 'MEDIUM'  # Multiple GT refs but same labels
        else:
            return 'LOW'  # Multiple GT refs with conflicting labels
    
    return 'UNRESOLVED'

def create_final_image_mapping(
    image_inventory: List[Dict],
    image_to_gt_map: Dict[str, List[Dict]],
    image_match_status: Dict[str, str],
    sha256_to_group: Dict[str, str]
) -> List[Dict]:
    """
    PART 5: Create Final Image-Level Mapping
    """
    print("PART 5: Creating Final Image-Level Mapping...")
    
    final_mapping = []
    
    for img in image_inventory:
        image_file = img['image_file']
        matches = image_to_gt_map.get(image_file, [])
        match_status = image_match_status[image_file]
        duplicate_group_id = sha256_to_group[img['sha256']]
        
        # Extract labels and metadata
        crops, stages, metadata = extract_labels_from_matches(matches)
        
        # Determine confidence
        confidence = determine_mapping_confidence(
            match_status, crops, stages, len(matches)
        )
        
        # Get GT reference (first one if multiple)
        gt_reference = matches[0]['gt_reference'] if matches else ''
        
        # Build row
        row = {
            'image_file': image_file,
            'relative_image_path': img['relative_image_path'],
            'filename': img['filename'],
            'sha256': img['sha256'],
            'duplicate_group_id': duplicate_group_id,
            'gt_match_status': match_status,
            'gt_reference': gt_reference,
            'gt_candidate_count': len(matches),
            'crop': ';'.join(sorted(crops)) if crops else '',
            'growth_stage': ';'.join(sorted(stages)) if stages else '',
            'Village_LG': metadata.get('Village_LG', ''),
            'district': metadata.get('district', ''),
            'taluk': metadata.get('taluk', ''),
            'village': metadata.get('village', ''),
            'survey_number': metadata.get('survey_number', ''),
            'sub_division': metadata.get('sub_division', ''),
            'final_lat': metadata.get('final_lat', ''),
            'final_long': metadata.get('final_long', ''),
            'timestamp': metadata.get('timestamp', ''),
            'image_timestamp': metadata.get('image_timestamp', ''),
            'image_latitude': metadata.get('image_latitude', ''),
            'image_longitude': metadata.get('image_longitude', ''),
            'mapping_confidence': confidence
        }
        
        final_mapping.append(row)
    
    print(f"Created {len(final_mapping)} final image mapping rows")
    return final_mapping

def create_duplicate_image_mapping(
    duplicate_groups: Dict[str, List[Dict]],
    image_to_gt_map: Dict[str, List[Dict]],
    sha256_to_group: Dict[str, str]
) -> List[Dict]:
    """
    PART 6: Create Duplicate Image Mapping
    """
    print("PART 6: Creating Duplicate Image Mapping...")
    
    duplicate_mapping = []
    
    for group_id, images in duplicate_groups.items():
        sha256 = images[0]['sha256']
        
        # Get labels for first image (all should have same if truly duplicates)
        first_image = images[0]
        matches = image_to_gt_map.get(first_image['image_file'], [])
        crops, stages, _ = extract_labels_from_matches(matches)
        
        # Determine match status
        match_status = 'MATCHED_UNIQUE' if len(matches) == 1 else 'MATCHED_MULTIPLE' if matches else 'NOT_FOUND'
        
        for img in images:
            matches = image_to_gt_map.get(img['image_file'], [])
            gt_reference = matches[0]['gt_reference'] if matches else ''
            
            row = {
                'duplicate_group_id': group_id,
                'sha256': sha256,
                'image_file': img['image_file'],
                'relative_image_path': img['relative_image_path'],
                'gt_reference': gt_reference,
                'crop': ';'.join(sorted(crops)) if crops else '',
                'growth_stage': ';'.join(sorted(stages)) if stages else '',
                'gt_match_status': match_status
            }
            duplicate_mapping.append(row)
    
    print(f"Created {len(duplicate_mapping)} duplicate mapping rows")
    return duplicate_mapping

def create_ambiguous_image_mapping(
    gt_to_images_map: Dict[str, List[str]],
    image_inventory: List[Dict],
    image_to_gt_map: Dict[str, List[Dict]],
    sha256_to_group: Dict[str, str]
) -> List[Dict]:
    """
    PART 7: Create Ambiguous Image Mapping
    """
    print("PART 7: Creating Ambiguous Image Mapping...")
    
    ambiguous_mapping = []
    
    # Find GT references with multiple image matches
    for gt_ref, image_files in gt_to_images_map.items():
        if len(image_files) > 1:
            # Get labels for this GT reference
            filename_to_image = {img['filename']: img for img in image_inventory}
            
            for image_file in image_files:
                img = filename_to_image.get(Path(image_file).name)
                if not img:
                    continue
                
                matches = image_to_gt_map.get(image_file, [])
                crops, stages, metadata = extract_labels_from_matches(matches)
                
                row = {
                    'gt_reference': gt_ref,
                    'candidate_image_file': image_file,
                    'candidate_sha256': img['sha256'],
                    'duplicate_group_id': sha256_to_group[img['sha256']],
                    'crop': ';'.join(sorted(crops)) if crops else '',
                    'growth_stage': ';'.join(sorted(stages)) if stages else '',
                    'image_timestamp': metadata.get('image_timestamp', ''),
                    'image_latitude': metadata.get('image_latitude', ''),
                    'image_longitude': metadata.get('image_longitude', ''),
                    'candidate_count': len(image_files),
                    'resolution_status': 'UNRESOLVED',
                    'notes': f'Multiple candidates ({len(image_files)}) for this GT reference'
                }
                ambiguous_mapping.append(row)
    
    print(f"Created {len(ambiguous_mapping)} ambiguous mapping rows")
    return ambiguous_mapping

def create_image_label_conflicts(
    image_inventory: List[Dict],
    image_to_gt_map: Dict[str, List[Dict]]
) -> Tuple[List[Dict], int, int]:
    """
    PART 8: Check for label conflicts
    Returns (conflicts, crop_conflict_count, stage_conflict_count)
    """
    print("PART 8: Checking for label conflicts...")
    
    conflicts = []
    crop_conflict_count = 0
    stage_conflict_count = 0
    
    for img in image_inventory:
        image_file = img['image_file']
        matches = image_to_gt_map.get(image_file, [])
        
        if not matches:
            continue
        
        crops, stages, _ = extract_labels_from_matches(matches)
        
        crop_labels = sorted(crops) if crops else []
        stage_labels = sorted(stages) if stages else []
        gt_references = [m['gt_reference'] for m in matches]
        
        conflict_type = []
        recommended_action = []
        
        if len(crop_labels) > 1:
            conflict_type.append('CROP_CONFLICT')
            recommended_action.append('Manual review required - multiple crop labels')
            crop_conflict_count += 1
        
        if len(stage_labels) > 1:
            conflict_type.append('STAGE_CONFLICT')
            recommended_action.append('Manual review required - multiple stage labels')
            stage_conflict_count += 1
        
        if conflict_type:
            row = {
                'image_file': image_file,
                'sha256': img['sha256'],
                'crop_labels': ';'.join(crop_labels),
                'stage_labels': ';'.join(stage_labels),
                'gt_references': ';'.join(gt_references),
                'conflict_type': ';'.join(conflict_type),
                'recommended_action': '; '.join(recommended_action)
            }
            conflicts.append(row)
    
    print(f"Crop conflicts: {crop_conflict_count}")
    print(f"Stage conflicts: {stage_conflict_count}")
    
    return conflicts, crop_conflict_count, stage_conflict_count

def create_mapping_summary(
    image_inventory: List[Dict],
    gt_rows: List[Dict],
    image_references_map: Dict[str, List[Dict]],
    image_match_status: Dict[str, str],
    sha256_to_group: Dict[str, str],
    duplicate_groups: Dict[str, List[Dict]],
    crop_conflict_count: int,
    stage_conflict_count: int,
    final_mapping: List[Dict]
) -> Dict[str, int]:
    """
    PART 9: Create Mapping Summary
    """
    print("PART 9: Creating Mapping Summary...")
    
    total_actual_images = len(image_inventory)
    total_ground_truth_rows = len(gt_rows)
    total_unique_gt_references = len(image_references_map)
    
    unique_image_matches = sum(1 for status in image_match_status.values() if status == 'MATCHED_UNIQUE')
    multiple_image_matches = sum(1 for status in image_match_status.values() if status == 'MATCHED_MULTIPLE')
    not_found_matches = sum(1 for status in image_match_status.values() if status == 'NOT_FOUND')
    
    unique_sha256_hashes = len(set(img['sha256'] for img in image_inventory))
    duplicate_groups_count = len(duplicate_groups)
    duplicate_image_files = sum(len(images) for images in duplicate_groups.values())
    
    high_confidence = sum(1 for row in final_mapping if row['mapping_confidence'] == 'HIGH')
    medium_confidence = sum(1 for row in final_mapping if row['mapping_confidence'] == 'MEDIUM')
    low_confidence = sum(1 for row in final_mapping if row['mapping_confidence'] == 'LOW')
    unresolved_mappings = sum(1 for row in final_mapping if row['mapping_confidence'] == 'UNRESOLVED')
    
    summary = {
        'total_actual_images': total_actual_images,
        'total_ground_truth_rows': total_ground_truth_rows,
        'total_unique_gt_references': total_unique_gt_references,
        'unique_image_matches': unique_image_matches,
        'multiple_image_matches': multiple_image_matches,
        'not_found_matches': not_found_matches,
        'unique_sha256_hashes': unique_sha256_hashes,
        'duplicate_groups': duplicate_groups_count,
        'duplicate_image_files': duplicate_image_files,
        'images_with_crop_conflicts': crop_conflict_count,
        'images_with_stage_conflicts': stage_conflict_count,
        'high_confidence_mappings': high_confidence,
        'medium_confidence_mappings': medium_confidence,
        'low_confidence_mappings': low_confidence,
        'unresolved_mappings': unresolved_mappings
    }
    
    print("Summary Statistics:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    return summary

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

def create_readme(summary: Dict[str, int], output_dir: Path) -> None:
    """Create README.md file"""
    readme_content = f"""# Final Dataset Mapping

## Overview
This directory contains the comprehensive mapping between actual image files and Ground Truth data, based on the corrected audit logic from dataset_audit_v2.

## Files Generated

### 1. final_image_mapping.csv
One row per actual image file with complete mapping information including:
- Image file details (path, filename, SHA-256)
- Ground Truth match status and reference
- Crop and growth stage labels
- Geographic and temporal metadata
- Mapping confidence level

### 2. duplicate_image_mapping.csv
Maps duplicate image groups identified by SHA-256:
- Group ID and hash
- All files in the group
- Ground Truth labels for comparison

### 3. ambiguous_image_mapping.csv
Images associated with ambiguous Ground Truth references:
- GT reference with multiple candidate images
- Resolution status and notes

### 4. image_label_conflicts.csv
Images with conflicting labels:
- Multiple crop or growth stage labels
- Recommended actions for resolution

### 5. mapping_summary.csv
Overall statistics about the dataset mapping

## Summary Statistics

- Total actual images: {summary['total_actual_images']}
- Total GT rows: {summary['total_ground_truth_rows']}
- Unique GT references: {summary['total_unique_gt_references']}

- Unique image mappings: {summary['unique_image_matches']}
- Multiple image mappings: {summary['multiple_image_matches']}
- Not found: {summary['not_found_matches']}

- Unique SHA-256 hashes: {summary['unique_sha256_hashes']}
- Duplicate groups: {summary['duplicate_groups']}
- Images in duplicate groups: {summary['duplicate_image_files']}

- Crop conflicts: {summary['images_with_crop_conflicts']}
- Stage conflicts: {summary['images_with_stage_conflicts']}

- HIGH confidence: {summary['high_confidence_mappings']}
- MEDIUM confidence: {summary['medium_confidence_mappings']}
- LOW confidence: {summary['low_confidence_mappings']}
- UNRESOLVED: {summary['unresolved_mappings']}

## Mapping Confidence Levels

- **HIGH**: One actual image maps uniquely to one Ground Truth reference with no conflicting labels
- **MEDIUM**: Plausible mapping with some ambiguity that does not create label conflicts
- **LOW**: Questionable or indirect association, or label conflicts present
- **UNRESOLVED**: No safe way to determine correct Ground Truth association

## Important Notes

- SHA-256 is used ONLY as a file-content fingerprint to identify exact duplicates
- SHA-256 is NOT used for Ground Truth label matching
- Ground Truth matching uses the corrected logic from dataset_audit_v2
- No original files were modified during this process
- No labels were invented or changed
- This mapping does NOT proceed to model training

## Data Safety

✓ No original images were deleted or renamed
✓ No Ground Truth CSV was modified
✓ No original audit files were overwritten
✓ No disease labels were invented
✓ No model training was performed
✓ No synthetic labels were created

Generated on: {summary.get('generated_date', 'N/A')}
"""
    
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"Created README.md at {readme_path}")

def create_output_zip(output_dir: Path) -> Path:
    """Create zip file of output directory"""
    zip_path = output_dir.parent / f"{output_dir.name}.zip"
    
    print(f"Creating zip file: {zip_path}")
    shutil.make_archive(str(output_dir), 'zip', str(output_dir.parent), output_dir.name)
    
    print(f"Zip file created: {zip_path}")
    return zip_path

def main():
    """Main execution function"""
    print("=" * 80)
    print("FINAL DATASET MAPPING GENERATION")
    print("=" * 80)
    print()
    
    # PART 1: Inventory images
    image_inventory = inventory_images(IMAGES_DIR)
    
    # PART 2: Parse Ground Truth
    gt_rows, image_references_map = parse_ground_truth(GROUND_TRUTH_CSV)
    
    # Load corrected matching from audit v2
    corrected_matching = load_corrected_matching(AUDIT_V2_DIR)
    
    # PART 3: Match images to Ground Truth
    image_to_gt_map, gt_to_images_map, image_match_status = match_images_to_ground_truth(
        image_inventory, image_references_map, corrected_matching
    )
    
    # PART 4: Create duplicate mapping
    sha256_to_group, duplicate_groups = create_duplicate_mapping(image_inventory)
    
    # PART 5: Create final image mapping
    final_mapping = create_final_image_mapping(
        image_inventory, image_to_gt_map, image_match_status, sha256_to_group
    )
    
    # PART 6: Create duplicate image mapping
    duplicate_mapping = create_duplicate_image_mapping(
        duplicate_groups, image_to_gt_map, sha256_to_group
    )
    
    # PART 7: Create ambiguous image mapping
    ambiguous_mapping = create_ambiguous_image_mapping(
        gt_to_images_map, image_inventory, image_to_gt_map, sha256_to_group
    )
    
    # PART 8: Check for conflicts
    conflicts, crop_conflict_count, stage_conflict_count = create_image_label_conflicts(
        image_inventory, image_to_gt_map
    )
    
    # PART 9: Create summary
    summary = create_mapping_summary(
        image_inventory, gt_rows, image_references_map, image_match_status,
        sha256_to_group, duplicate_groups, crop_conflict_count, stage_conflict_count,
        final_mapping
    )
    
    # PART 10: Write all output files
    print("\nPART 10: Writing output files...")
    
    write_csv(final_mapping, OUTPUT_DIR / "final_image_mapping.csv")
    write_csv(duplicate_mapping, OUTPUT_DIR / "duplicate_image_mapping.csv")
    write_csv(ambiguous_mapping, OUTPUT_DIR / "ambiguous_image_mapping.csv")
    write_csv(conflicts, OUTPUT_DIR / "image_label_conflicts.csv")
    
    # Write summary as CSV
    summary_rows = [{'metric': k, 'value': v} for k, v in summary.items()]
    write_csv(summary_rows, OUTPUT_DIR / "mapping_summary.csv")
    
    # Create README
    create_readme(summary, OUTPUT_DIR)
    
    # Create zip file
    zip_path = create_output_zip(OUTPUT_DIR)
    
    # Print final console output
    print("\n" + "=" * 80)
    print("FINAL CONSOLE OUTPUT")
    print("=" * 80)
    print(f"Total actual images: {summary['total_actual_images']}")
    print(f"Total GT rows: {summary['total_ground_truth_rows']}")
    print(f"Unique GT references: {summary['total_unique_gt_references']}")
    print()
    print(f"Unique image mappings: {summary['unique_image_matches']}")
    print(f"Ambiguous mappings: {summary['multiple_image_matches']}")
    print(f"Not found: {summary['not_found_matches']}")
    print()
    print(f"Unique SHA-256 hashes: {summary['unique_sha256_hashes']}")
    print(f"Duplicate groups: {summary['duplicate_groups']}")
    print(f"Images in duplicate groups: {summary['duplicate_image_files']}")
    print()
    print(f"Crop conflicts: {summary['images_with_crop_conflicts']}")
    print(f"Stage conflicts: {summary['images_with_stage_conflicts']}")
    print()
    print(f"HIGH confidence: {summary['high_confidence_mappings']}")
    print(f"MEDIUM confidence: {summary['medium_confidence_mappings']}")
    print(f"LOW confidence: {summary['low_confidence_mappings']}")
    print(f"UNRESOLVED: {summary['unresolved_mappings']}")
    print()
    print("Mapping completed. No original dataset files were modified.")
    print("=" * 80)

if __name__ == "__main__":
    main()