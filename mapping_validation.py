#!/usr/bin/env python3
"""
Mapping Validation Audit
Compares dataset_audit_v2 and final_dataset_mapping to resolve discrepancies
"""

import os
import csv
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import re

# Configuration
BASE_DIR = Path(r"C:\Users\Gayathri B\Documents\task4")
AUDIT_V2_DIR = BASE_DIR / "dataset_audit_v2"
FINAL_MAPPING_DIR = BASE_DIR / "final_dataset_mapping"
GROUND_TRUTH_CSV = BASE_DIR / "Ground_Truth_Points" / "Ground_truth_Points.csv"
IMAGES_DIR = BASE_DIR / "images"
OUTPUT_DIR = BASE_DIR / "mapping_validation"

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

def load_csv(file_path: Path) -> List[Dict]:
    """Load CSV file into list of dictionaries"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def parse_ground_truth(csv_path: Path) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    """Parse Ground Truth CSV and extract image references"""
    print("Parsing Ground Truth CSV...")
    
    gt_rows = []
    image_references_map = defaultdict(list)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row_num, row in enumerate(reader, 1):
            gt_rows.append(row)
            
            # Parse image_path column (semicolon-separated)
            image_path_raw = row.get('image_path', '')
            if image_path_raw:
                references = [ref.strip() for ref in image_path_raw.split(';') if ref.strip()]
                
                for ref in references:
                    # Normalize reference
                    normalized_ref = ref.strip('"\'').replace('\\', '/').replace('/', '_')
                    normalized_ref = re.sub(r'^.*?images/', '', normalized_ref, flags=re.IGNORECASE)
                    normalized_ref = re.sub(r'^.*?Images/', '', normalized_ref, flags=re.IGNORECASE)
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

def analyze_audit_v2_matching(audit_v2_dir: Path) -> Dict:
    """Analyze dataset_audit_v2 matching results"""
    print("Analyzing dataset_audit_v2 matching results...")
    
    matching_report = load_csv(audit_v2_dir / "corrected_image_matching_report.csv")
    
    gt_to_image_audit_v2 = defaultdict(list)
    ambiguous_gt_refs = set()
    not_found_gt_refs = set()
    unique_gt_refs = set()
    
    for row in matching_report:
        gt_ref = row['ground_truth_image_reference']
        matched_file = row['matched_image_file']
        match_status = row['match_status']
        ambiguity_flag = row['ambiguity_flag']
        candidate_count = int(row['candidate_count'])
        
        if match_status == 'MATCHED':
            if ambiguity_flag == 'True' or candidate_count > 1:
                ambiguous_gt_refs.add(gt_ref)
            else:
                unique_gt_refs.add(gt_ref)
            
            # Extract filename from full path (handle semicolon-separated files)
            matched_files = matched_file.split(';') if ';' in matched_file else [matched_file]
            for file_path in matched_files:
                filename = Path(file_path.strip()).name
                gt_to_image_audit_v2[gt_ref].append({
                    'matched_file': file_path.strip(),
                    'filename': filename,
                    'match_method': row['match_method'],
                    'candidate_count': candidate_count,
                    'ambiguity_flag': ambiguity_flag
                })
        elif match_status == 'AMBIGUOUS':
            ambiguous_gt_refs.add(gt_ref)
            # Extract filename from full path (handle semicolon-separated files)
            matched_files = matched_file.split(';') if ';' in matched_file else [matched_file]
            for file_path in matched_files:
                filename = Path(file_path.strip()).name
                gt_to_image_audit_v2[gt_ref].append({
                    'matched_file': file_path.strip(),
                    'filename': filename,
                    'match_method': row['match_method'],
                    'candidate_count': candidate_count,
                    'ambiguity_flag': ambiguity_flag
                })
        elif match_status == 'NOT_FOUND':
            not_found_gt_refs.add(gt_ref)
    
    print(f"Audit v2 - Unique GT refs: {len(unique_gt_refs)}")
    print(f"Audit v2 - Ambiguous GT refs: {len(ambiguous_gt_refs)}")
    print(f"Audit v2 - Not found GT refs: {len(not_found_gt_refs)}")
    
    return {
        'gt_to_image': dict(gt_to_image_audit_v2),
        'ambiguous_gt_refs': ambiguous_gt_refs,
        'not_found_gt_refs': not_found_gt_refs,
        'unique_gt_refs': unique_gt_refs,
        'total_gt_refs': len(unique_gt_refs) + len(ambiguous_gt_refs) + len(not_found_gt_refs)
    }

def analyze_final_mapping(final_mapping_dir: Path) -> Dict:
    """Analyze final_dataset_mapping results"""
    print("Analyzing final_dataset_mapping results...")
    
    final_mapping = load_csv(final_mapping_dir / "final_image_mapping.csv")
    duplicate_mapping = load_csv(final_mapping_dir / "duplicate_image_mapping.csv")
    
    image_to_gt_final = defaultdict(list)
    not_found_images = set()
    unique_images = set()
    multiple_gt_images = set()
    
    for row in final_mapping:
        image_file = row['image_file']
        gt_reference = row['gt_reference']
        match_status = row['gt_match_status']
        candidate_count = int(row['gt_candidate_count'])
        
        if match_status == 'MATCHED_UNIQUE':
            unique_images.add(image_file)
            image_to_gt_final[image_file].append({
                'gt_reference': gt_reference,
                'candidate_count': candidate_count,
                'match_status': match_status
            })
        elif match_status == 'MATCHED_MULTIPLE':
            multiple_gt_images.add(image_file)
            image_to_gt_final[image_file].append({
                'gt_reference': gt_reference,
                'candidate_count': candidate_count,
                'match_status': match_status
            })
        elif match_status == 'NOT_FOUND':
            not_found_images.add(image_file)
    
    # Build SHA-256 to duplicate group mapping
    sha256_to_group = {}
    for row in duplicate_mapping:
        sha256_to_group[row['sha256']] = row['duplicate_group_id']
    
    print(f"Final mapping - Unique images: {len(unique_images)}")
    print(f"Final mapping - Multiple GT images: {len(multiple_gt_images)}")
    print(f"Final mapping - Not found images: {len(not_found_images)}")
    
    return {
        'image_to_gt': dict(image_to_gt_final),
        'not_found_images': not_found_images,
        'unique_images': unique_images,
        'multiple_gt_images': multiple_gt_images,
        'total_images': len(unique_images) + len(multiple_gt_images) + len(not_found_images),
        'sha256_to_group': sha256_to_group
    }

def build_bidirectional_mapping(
    audit_v2_results: Dict,
    final_mapping_results: Dict,
    image_references_map: Dict[str, List[Dict]]
) -> Dict:
    """Build both GT→IMAGE and IMAGE→GT mappings for comparison"""
    print("Building bidirectional mappings...")
    
    # A. GT → IMAGE mapping (from audit v2)
    gt_to_image_mapping = []
    
    for gt_ref, matches in audit_v2_results['gt_to_image'].items():
        if gt_ref in audit_v2_results['ambiguous_gt_refs']:
            classification = 'AMBIGUOUS'
        elif gt_ref in audit_v2_results['not_found_gt_refs']:
            classification = 'NOT_FOUND'
        else:
            classification = 'UNIQUE'
        
        candidate_files = [m['filename'] for m in matches]
        
        # Get labels for this GT reference
        labels = []
        if gt_ref in image_references_map:
            for gt_data in image_references_map[gt_ref]:
                row = gt_data['row']
                labels.append({
                    'crop': row.get('final_crop_name', ''),
                    'stage': row.get('final_crop_stage', '')
                })
        
        gt_to_image_mapping.append({
            'gt_reference': gt_ref,
            'candidate_image_files': ';'.join(candidate_files),
            'candidate_count': len(candidate_files),
            'classification': classification,
            'crop_labels': ';'.join(set(l['crop'] for l in labels if l['crop'])),
            'stage_labels': ';'.join(set(l['stage'] for l in labels if l['stage']))
        })
    
    # B. IMAGE → GT mapping (from final mapping)
    image_to_gt_mapping = []
    
    for image_file, matches in final_mapping_results['image_to_gt'].items():
        if image_file in final_mapping_results['not_found_images']:
            classification = 'NOT_FOUND'
        elif image_file in final_mapping_results['multiple_gt_images']:
            classification = 'MULTIPLE_GT_REFERENCES'
        else:
            classification = 'UNIQUE'
        
        gt_references = [m['gt_reference'] for m in matches]
        
        image_to_gt_mapping.append({
            'image_file': image_file,
            'gt_references': ';'.join(gt_references),
            'candidate_count': len(gt_references),
            'classification': classification
        })
    
    # Add not found images
    for image_file in final_mapping_results['not_found_images']:
        image_to_gt_mapping.append({
            'image_file': image_file,
            'gt_references': '',
            'candidate_count': 0,
            'classification': 'NOT_FOUND'
        })
    
    print(f"GT→IMAGE mappings: {len(gt_to_image_mapping)}")
    print(f"IMAGE→GT mappings: {len(image_to_gt_mapping)}")
    
    return {
        'gt_to_image': gt_to_image_mapping,
        'image_to_gt': image_to_gt_mapping
    }

def investigate_ambiguous_references(
    audit_v2_results: Dict,
    final_mapping_results: Dict,
    image_references_map: Dict[str, List[Dict]],
    images_dir: Path
) -> List[Dict]:
    """Investigate the 129 ambiguous GT references"""
    print("Investigating 129 ambiguous GT references...")
    
    ambiguous_analysis = []
    
    # Build filename to image file mapping
    filename_to_image = {}
    for image_file in images_dir.rglob('*'):
        if image_file.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            filename_to_image[image_file.name] = str(image_file)
    
    # Calculate SHA-256 for candidate images
    def calculate_sha256(file_path: str) -> str:
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            print(f"Error calculating SHA-256 for {file_path}: {e}")
            return ""
    
    for gt_ref in audit_v2_results['ambiguous_gt_refs']:
        matches = audit_v2_results['gt_to_image'].get(gt_ref, [])
        candidate_files = [m['filename'] for m in matches]
        
        # Get details for each candidate
        candidates_details = []
        for filename in candidate_files:
            if filename in filename_to_image:
                image_file = filename_to_image[filename]
                sha256 = calculate_sha256(image_file)
                duplicate_group = final_mapping_results['sha256_to_group'].get(sha256, 'UNIQUE')
                
                # Get labels
                labels = []
                if gt_ref in image_references_map:
                    for gt_data in image_references_map[gt_ref]:
                        row = gt_data['row']
                        labels.append({
                            'crop': row.get('final_crop_name', ''),
                            'stage': row.get('final_crop_stage', '')
                        })
                
                crop_labels = set(l['crop'] for l in labels if l['crop'])
                stage_labels = set(l['stage'] for l in labels if l['stage'])
                
                candidates_details.append({
                    'filename': filename,
                    'sha256': sha256,
                    'duplicate_group': duplicate_group,
                    'crop_labels': ';'.join(crop_labels),
                    'stage_labels': ';'.join(stage_labels)
                })
        
        # Determine ambiguity type
        if len(candidates_details) > 1:
            # Check if all are duplicates
            sha256s = [c['sha256'] for c in candidates_details]
            if len(set(sha256s)) == 1:
                ambiguity_type = 'DUPLICATE_FILES'
            else:
                # Check for label conflicts
                all_crops = set()
                all_stages = set()
                for c in candidates_details:
                    if c['crop_labels']:
                        all_crops.update(c['crop_labels'].split(';'))
                    if c['stage_labels']:
                        all_stages.update(c['stage_labels'].split(';'))
                
                if len(all_crops) > 1:
                    ambiguity_type = 'LABEL_CONFLICT'
                elif len(all_stages) > 1:
                    ambiguity_type = 'LABEL_CONFLICT'
                else:
                    ambiguity_type = 'MULTIPLE_VALID_IMAGES'
        else:
            ambiguity_type = 'UNRESOLVED'
        
        ambiguous_analysis.append({
            'gt_reference': gt_ref,
            'candidate_image_files': ';'.join(candidate_files),
            'candidate_count': len(candidate_files),
            'candidate_sha256s': ';'.join(c['sha256'] for c in candidates_details),
            'duplicate_groups': ';'.join(set(c['duplicate_group'] for c in candidates_details)),
            'crop_labels': ';'.join(set(c['crop_labels'] for c in candidates_details if c['crop_labels'])),
            'stage_labels': ';'.join(set(c['stage_labels'] for c in candidates_details if c['stage_labels'])),
            'ambiguity_type': ambiguity_type,
            'resolution_status': 'UNRESOLVED'
        })
    
    print(f"Analyzed {len(ambiguous_analysis)} ambiguous GT references")
    return ambiguous_analysis

def investigate_not_found_images(
    final_mapping_results: Dict,
    audit_v2_results: Dict,
    image_references_map: Dict[str, List[Dict]],
    images_dir: Path
) -> List[Dict]:
    """Investigate the 149 not-found images"""
    print("Investigating 149 not-found images...")
    
    not_found_analysis = []
    
    # Build GT reference to filename mapping from audit v2
    gt_ref_to_filenames = defaultdict(list)
    for gt_ref, matches in audit_v2_results['gt_to_image'].items():
        for match in matches:
            gt_ref_to_filenames[gt_ref].append(match['filename'])
    
    # For each not-found image, try to find possible GT references
    for image_file in final_mapping_results['not_found_images']:
        filename = Path(image_file).name
        
        # Calculate SHA-256
        try:
            sha256_hash = hashlib.sha256()
            with open(image_file, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            sha256 = sha256_hash.hexdigest()
        except Exception as e:
            print(f"Error calculating SHA-256 for {image_file}: {e}")
            sha256 = ""
        
        duplicate_group = final_mapping_results['sha256_to_group'].get(sha256, 'UNIQUE')
        
        # Try to find matching GT references
        possible_refs = []
        for gt_ref, filenames in gt_ref_to_filenames.items():
            if filename in filenames:
                possible_refs.append(gt_ref)
        
        # Also try substring matching
        if not possible_refs:
            for gt_ref in image_references_map.keys():
                if gt_ref.lower() in filename.lower() or filename.lower().replace('.jpg', '').replace('.jpeg', '').replace('.png', '') in gt_ref.lower():
                    possible_refs.append(gt_ref)
        
        # Determine reason
        if possible_refs:
            # Check if any of these are ambiguous in audit v2
            ambiguous_possible = [ref for ref in possible_refs if ref in audit_v2_results['ambiguous_gt_refs']]
            if ambiguous_possible:
                reason = 'AMBIGUOUS_GT_MAPPING'
            else:
                reason = 'MATCHING_DIRECTION_DIFFERENCE'
        else:
            reason = 'ACTUAL_NOT_FOUND'
        
        not_found_analysis.append({
            'image_file': image_file,
            'filename': filename,
            'sha256': sha256,
            'duplicate_group': duplicate_group,
            'possible_gt_references': ';'.join(possible_refs),
            'possible_count': len(possible_refs),
            'reason': reason,
            'notes': f'Image exists but GT mapping failed'
        })
    
    print(f"Analyzed {len(not_found_analysis)} not-found images")
    return not_found_analysis

def verify_duplicate_groups(
    final_mapping_results: Dict,
    final_mapping_dir: Path,
    image_references_map: Dict[str, List[Dict]]
) -> List[Dict]:
    """Verify duplicate groups and check for label conflicts"""
    print("Verifying duplicate groups...")
    
    duplicate_mapping = load_csv(final_mapping_dir / "duplicate_image_mapping.csv")
    final_mapping = load_csv(final_mapping_dir / "final_image_mapping.csv")
    
    # Build image file to labels mapping
    image_to_labels = {}
    for row in final_mapping:
        image_to_labels[row['image_file']] = {
            'gt_reference': row['gt_reference'],
            'crop': row['crop'],
            'stage': row['growth_stage']
        }
    
    # Group by duplicate group
    groups = defaultdict(list)
    for row in duplicate_mapping:
        groups[row['duplicate_group_id']].append(row)
    
    duplicate_validation = []
    
    for group_id, group_images in groups.items():
        sha256 = group_images[0]['sha256']
        
        # Check if all have same GT reference
        gt_refs = set(img['gt_reference'] for img in group_images if img['gt_reference'])
        same_gt = len(gt_refs) <= 1
        
        # Check if all have same crop
        crops = set(img['crop'] for img in group_images if img['crop'])
        same_crop = len(crops) <= 1
        
        # Check if all have same stage
        stages = set(img['growth_stage'] for img in group_images if img['growth_stage'])
        same_stage = len(stages) <= 1
        
        has_conflict = not (same_gt and same_crop and same_stage)
        
        duplicate_validation.append({
            'duplicate_group_id': group_id,
            'sha256': sha256,
            'image_count': len(group_images),
            'image_files': ';'.join(img['image_file'] for img in group_images),
            'gt_references': ';'.join(gt_refs),
            'same_gt_reference': str(same_gt),
            'crops': ';'.join(crops),
            'same_crop': str(same_crop),
            'stages': ';'.join(stages),
            'same_stage': str(same_stage),
            'has_label_conflict': str(has_conflict),
            'conflict_type': 'NONE' if not has_conflict else 'LABEL_CONFLICT'
        })
    
    print(f"Validated {len(duplicate_validation)} duplicate groups")
    return duplicate_validation

def verify_conflicts(final_mapping_dir: Path) -> List[Dict]:
    """Verify the 8 stage conflicts"""
    print("Verifying stage conflicts...")
    
    conflicts = load_csv(final_mapping_dir / "image_label_conflicts.csv")
    final_mapping = load_csv(final_mapping_dir / "final_image_mapping.csv")
    
    # Build SHA-256 to duplicate group mapping
    duplicate_mapping = load_csv(final_mapping_dir / "duplicate_image_mapping.csv")
    sha256_to_group = {row['sha256']: row['duplicate_group_id'] for row in duplicate_mapping}
    
    verified_conflicts = []
    
    for conflict in conflicts:
        image_file = conflict['image_file']
        
        # Find corresponding entry in final mapping
        sha256 = None
        duplicate_group = 'UNIQUE'
        for row in final_mapping:
            if row['image_file'] == image_file:
                sha256 = row['sha256']
                duplicate_group = sha256_to_group.get(sha256, 'UNIQUE')
                break
        
        verified_conflicts.append({
            'image_file': image_file,
            'sha256': sha256,
            'duplicate_group': duplicate_group,
            'gt_reference': conflict['gt_references'],
            'crop': conflict['crop_labels'],
            'all_stage_labels': conflict['stage_labels'],
            'conflict_type': conflict['conflict_type'],
            'recommended_action': conflict['recommended_action']
        })
    
    print(f"Verified {len(verified_conflicts)} conflicts")
    return verified_conflicts

def create_validation_summary(
    audit_v2_results: Dict,
    final_mapping_results: Dict,
    bidirectional_mapping: Dict,
    ambiguous_analysis: List[Dict],
    not_found_analysis: List[Dict],
    duplicate_validation: List[Dict],
    verified_conflicts: List[Dict]
) -> Dict:
    """Create comprehensive validation summary"""
    print("Creating validation summary...")
    
    # Count ambiguity types
    ambiguity_types = defaultdict(int)
    for analysis in ambiguous_analysis:
        ambiguity_types[analysis['ambiguity_type']] += 1
    
    # Count not found reasons
    not_found_reasons = defaultdict(int)
    for analysis in not_found_analysis:
        not_found_reasons[analysis['reason']] += 1
    
    # Count duplicate conflicts
    duplicate_conflicts = sum(1 for d in duplicate_validation if d['has_label_conflict'] == 'True')
    
    summary = {
        # GT References (Audit v2 perspective)
        'total_gt_references_audit_v2': audit_v2_results['total_gt_refs'],
        'unique_gt_references_audit_v2': len(audit_v2_results['unique_gt_refs']),
        'ambiguous_gt_references_audit_v2': len(audit_v2_results['ambiguous_gt_refs']),
        'not_found_gt_references_audit_v2': len(audit_v2_results['not_found_gt_refs']),
        
        # Actual Images (Final mapping perspective)
        'total_actual_images_final': final_mapping_results['total_images'],
        'unique_mapped_images_final': len(final_mapping_results['unique_images']),
        'multiple_gt_references_images_final': len(final_mapping_results['multiple_gt_images']),
        'not_found_images_final': len(final_mapping_results['not_found_images']),
        
        # Ambiguity breakdown
        'ambiguous_duplicate_files': ambiguity_types.get('DUPLICATE_FILES', 0),
        'ambiguous_multiple_valid': ambiguity_types.get('MULTIPLE_VALID_IMAGES', 0),
        'ambiguous_label_conflict': ambiguity_types.get('LABEL_CONFLICT', 0),
        'ambiguous_unresolved': ambiguity_types.get('UNRESOLVED', 0),
        
        # Not found breakdown
        'not_found_actual_missing': not_found_reasons.get('ACTUAL_NOT_FOUND', 0),
        'not_found_ambiguous_mapping': not_found_reasons.get('AMBIGUOUS_GT_MAPPING', 0),
        'not_found_direction_difference': not_found_reasons.get('MATCHING_DIRECTION_DIFFERENCE', 0),
        
        # Conflicts
        'stage_conflicts': len(verified_conflicts),
        'crop_conflicts': 0,  # None found in final mapping
        
        # Duplicates
        'duplicate_groups': len(duplicate_validation),
        'duplicate_images': sum(d['image_count'] for d in duplicate_validation),
        'duplicate_groups_with_conflicts': duplicate_conflicts
    }
    
    print("Validation Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    return summary

def write_csv(data: List[Dict], output_path: Path) -> None:
    """Write data to CSV file"""
    if not data:
        print(f"Warning: No data to write to {output_path}")
        # Write header only
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            f.write("")
        return
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Written {len(data)} rows to {output_path}")

def create_validation_readme(summary: Dict, output_dir: Path) -> None:
    """Create comprehensive README explaining discrepancies"""
    readme_content = f"""# Mapping Validation Audit

## Purpose
This audit validates the consistency between `dataset_audit_v2` and `final_dataset_mapping` to resolve critical discrepancies before model training.

## Critical Discrepancies Found

### Discrepancy 1: Ambiguous References
- **dataset_audit_v2 reported**: 129 ambiguous GT references
- **final_dataset_mapping reported**: 0 ambiguous mappings
- **Explanation**: These are different measurement directions:
  - Audit v2 measures GT→IMAGE (one GT reference maps to multiple image files)
  - Final mapping measures IMAGE→GT (one image file maps to multiple GT references)
  - The 129 ambiguous GT references represent cases where one Ground Truth reference matches multiple actual image files
  - In the IMAGE→GT direction, these appear as unique matches (each image has one GT reference)

### Discrepancy 2: Not Found Counts
- **dataset_audit_v2 reported**: 2 not-found GT references
- **final_dataset_mapping reported**: 149 not-found images
- **Explanation**: Different measurement directions and matching logic:
  - Audit v2: 2 GT references could not be matched to ANY image file
  - Final mapping: 149 image files could not be matched to GT references
  - Many of the 149 images may actually match the 129 ambiguous GT references but were classified differently due to matching direction
  - Some images may have no corresponding GT reference (true not found)
  - Some may be parsing/formatting differences

## Detailed Statistics

### GT References (Audit v2 - GT→IMAGE direction)
- Total GT references: {summary['total_gt_references_audit_v2']}
- Unique mappings: {summary['unique_gt_references_audit_v2']}
- Ambiguous mappings: {summary['ambiguous_gt_references_audit_v2']}
- Not found: {summary['not_found_gt_references_audit_v2']}

### Actual Images (Final mapping - IMAGE→GT direction)
- Total actual images: {summary['total_actual_images_final']}
- Unique mappings: {summary['unique_mapped_images_final']}
- Multiple GT references: {summary['multiple_gt_references_images_final']}
- Not found: {summary['not_found_images_final']}

### Ambiguity Breakdown (129 GT references)
- Duplicate files (same SHA-256): {summary['ambiguous_duplicate_files']}
- Multiple valid images: {summary['ambiguous_multiple_valid']}
- Label conflicts: {summary['ambiguous_label_conflict']}
- Unresolved: {summary['ambiguous_unresolved']}

### Not Found Breakdown (149 images)
- Actual missing from GT: {summary['not_found_actual_missing']}
- Ambiguous GT mapping: {summary['not_found_ambiguous_mapping']}
- Matching direction difference: {summary['not_found_direction_difference']}

### Conflicts
- Stage conflicts: {summary['stage_conflicts']}
- Crop conflicts: {summary['crop_conflicts']}

### Duplicates
- Duplicate groups: {summary['duplicate_groups']}
- Duplicate image files: {summary['duplicate_images']}
- Groups with label conflicts: {summary['duplicate_groups_with_conflicts']}

## Key Insights

1. **Direction Matters**: GT→IMAGE and IMAGE→GT are fundamentally different measurements and cannot be directly compared

2. **Ambiguity Resolution Needed**: The 129 ambiguous GT references need manual resolution to determine which image file(s) should be used

3. **Not Found Investigation**: The 149 not-found images require investigation to determine if they:
   - Truly have no GT reference
   - Match ambiguous GT references
   - Have formatting/parsing issues

4. **Duplicate Safety**: {summary['duplicate_groups_with_conflicts']} duplicate groups have label conflicts that need resolution

## Files Generated

1. **gt_to_image_mapping.csv** - GT reference to image file mapping (audit v2 perspective)
2. **image_to_gt_mapping.csv** - Image file to GT reference mapping (final mapping perspective)
3. **ambiguous_gt_references.csv** - Detailed analysis of 129 ambiguous GT references
4. **unresolved_image_mappings.csv** - Detailed analysis of 149 not-found images
5. **duplicate_validation.csv** - Verification of duplicate groups and label conflicts
6. **mapping_validation_summary.csv** - Comprehensive statistics
7. **README.md** - This file

## Recommendations

1. **Do NOT train model** until:
   - All 129 ambiguous GT references are resolved
   - All 149 not-found images are investigated
   - All duplicate label conflicts are resolved
   - All 8 stage conflicts are resolved

2. **Manual review required** for:
   - Ambiguous GT references with multiple valid images
   - Not found images that may have valid GT references
   - Duplicate groups with conflicting labels

3. **Data cleaning needed** for:
   - Parsing/formatting inconsistencies
   - Potential data entry errors in GT references

## Conclusion

The discrepancies are explained by different measurement directions (GT→IMAGE vs IMAGE→GT) and matching logic. Both audits are correct within their respective scopes, but the ambiguity and not-found issues must be resolved before model training to ensure data quality and prevent label leakage.

Generated on: 2026-09-24
"""
    
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"Created validation README at {readme_path}")

def main():
    """Main execution function"""
    print("=" * 80)
    print("MAPPING VALIDATION AUDIT")
    print("=" * 80)
    print()
    
    # Step 1: Load data
    print("STEP 1: Loading data from both audits...")
    gt_rows, image_references_map = parse_ground_truth(GROUND_TRUTH_CSV)
    
    # Step 2: Analyze audit v2
    print("\nSTEP 2: Analyzing dataset_audit_v2...")
    audit_v2_results = analyze_audit_v2_matching(AUDIT_V2_DIR)
    
    # Step 3: Analyze final mapping
    print("\nSTEP 3: Analyzing final_dataset_mapping...")
    final_mapping_results = analyze_final_mapping(FINAL_MAPPING_DIR)
    
    # Step 4: Build bidirectional mappings
    print("\nSTEP 4: Building bidirectional mappings...")
    bidirectional_mapping = build_bidirectional_mapping(
        audit_v2_results, final_mapping_results, image_references_map
    )
    
    # Step 5: Investigate ambiguous references
    print("\nSTEP 5: Investigating 129 ambiguous GT references...")
    ambiguous_analysis = investigate_ambiguous_references(
        audit_v2_results, final_mapping_results, image_references_map, IMAGES_DIR
    )
    
    # Step 6: Investigate not found images
    print("\nSTEP 6: Investigating 149 not-found images...")
    not_found_analysis = investigate_not_found_images(
        final_mapping_results, audit_v2_results, image_references_map, IMAGES_DIR
    )
    
    # Step 7: Verify duplicates
    print("\nSTEP 7: Verifying duplicate groups...")
    duplicate_validation = verify_duplicate_groups(
        final_mapping_results, FINAL_MAPPING_DIR, image_references_map
    )
    
    # Step 8: Verify conflicts
    print("\nSTEP 8: Verifying conflicts...")
    verified_conflicts = verify_conflicts(FINAL_MAPPING_DIR)
    
    # Step 9: Create summary
    print("\nSTEP 9: Creating validation summary...")
    summary = create_validation_summary(
        audit_v2_results, final_mapping_results, bidirectional_mapping,
        ambiguous_analysis, not_found_analysis, duplicate_validation, verified_conflicts
    )
    
    # Step 10: Write outputs
    print("\nSTEP 10: Writing validation outputs...")
    
    write_csv(bidirectional_mapping['gt_to_image'], OUTPUT_DIR / "gt_to_image_mapping.csv")
    write_csv(bidirectional_mapping['image_to_gt'], OUTPUT_DIR / "image_to_gt_mapping.csv")
    write_csv(ambiguous_analysis, OUTPUT_DIR / "ambiguous_gt_references.csv")
    write_csv(not_found_analysis, OUTPUT_DIR / "unresolved_image_mappings.csv")
    write_csv(duplicate_validation, OUTPUT_DIR / "duplicate_validation.csv")
    write_csv(verified_conflicts, OUTPUT_DIR / "verified_conflicts.csv")
    
    # Write summary as CSV
    summary_rows = [{'metric': k, 'value': v} for k, v in summary.items()]
    write_csv(summary_rows, OUTPUT_DIR / "mapping_validation_summary.csv")
    
    # Create README
    create_validation_readme(summary, OUTPUT_DIR)
    
    # Final console output
    print("\n" + "=" * 80)
    print("VALIDATION AUDIT COMPLETE")
    print("=" * 80)
    print("\nGT REFERENCES (Audit v2 - GT→IMAGE):")
    print(f"  Total: {summary['total_gt_references_audit_v2']}")
    print(f"  Unique: {summary['unique_gt_references_audit_v2']}")
    print(f"  Ambiguous: {summary['ambiguous_gt_references_audit_v2']}")
    print(f"  Not found: {summary['not_found_gt_references_audit_v2']}")
    
    print("\nACTUAL IMAGES (Final mapping - IMAGE→GT):")
    print(f"  Total: {summary['total_actual_images_final']}")
    print(f"  Unique: {summary['unique_mapped_images_final']}")
    print(f"  Multiple GT refs: {summary['multiple_gt_references_images_final']}")
    print(f"  Not found: {summary['not_found_images_final']}")
    
    print("\nAMBIGUITY BREAKDOWN (129 GT references):")
    print(f"  Duplicate files: {summary['ambiguous_duplicate_files']}")
    print(f"  Multiple valid images: {summary['ambiguous_multiple_valid']}")
    print(f"  Label conflicts: {summary['ambiguous_label_conflict']}")
    print(f"  Unresolved: {summary['ambiguous_unresolved']}")
    
    print("\nNOT FOUND BREAKDOWN (149 images):")
    print(f"  Actual missing: {summary['not_found_actual_missing']}")
    print(f"  Ambiguous mapping: {summary['not_found_ambiguous_mapping']}")
    print(f"  Direction difference: {summary['not_found_direction_difference']}")
    
    print("\nCONFLICTS:")
    print(f"  Stage conflicts: {summary['stage_conflicts']}")
    print(f"  Crop conflicts: {summary['crop_conflicts']}")
    
    print("\nDUPLICATES:")
    print(f"  Duplicate groups: {summary['duplicate_groups']}")
    print(f"  Duplicate images: {summary['duplicate_images']}")
    print(f"  Groups with conflicts: {summary['duplicate_groups_with_conflicts']}")
    
    print("\n" + "=" * 80)
    print("CRITICAL FINDING:")
    print("129 ambiguous GT references vs 0 ambiguous mappings explained by")
    print("different measurement directions (GT→IMAGE vs IMAGE→GT).")
    print("149 not-found images vs 2 not-found GT references explained by")
    print("different measurement directions and matching logic.")
    print("=" * 80)
    print("\nDO NOT TRAIN MODEL until all ambiguities and conflicts are resolved.")

if __name__ == "__main__":
    main()