import os
import sys
import json
import random
import re
import torch
from torch.utils.data import DataLoader

try:
    from dataset import FmriDataset, collate_fn_skip_error
except ImportError:
    print("Error: Make sure dataset.py is in the same directory.")
    sys.exit(1)


def extract_id_from_filename(filename):
    """
    Extracts the leading numerical ID from filenames like '000300655084_fALFF.nii'.
    """
    match = re.match(r'^(\d+)_', filename) #regex
    return match.group(1) if match else None


def prepare_dataloaders(config, wandb_logger=None):
    """
    Loads label map, processes labels, splits data, creates Datasets & DataLoaders.

    Args:
        config (dict): The loaded configuration dictionary.
        wandb_logger (WandbLogger, optional): Logger instance for logging info.

    Returns:
        tuple: (trainloader, testloader) or (None, None) on failure.
    """
    # --- Extract Configuration ---
    try:
        label_map_path = config['data']['label_map_path']
        fmri_dir = config['data']['fmri_dir']
        label_str_to_int = config['dataset']['label_map']
        target_shape = config['dataset']['target_shape']

        # Additional keys for preprocessing (with defaults)
        use_time_slices    = config['dataset'].get('use_time_slices', False)
        apply_quantile_norm = config['dataset'].get('apply_quantile_norm', True)
        apply_crop         = config['dataset'].get('apply_crop', False)
        crop_percentile    = config['dataset'].get('crop_percentile', 10)

        loader_cfg   = config.get('dataloader', {})
        batch_size   = loader_cfg.get('batch_size', 8)
        num_workers  = loader_cfg.get('num_workers', 4)
        split_ratio  = loader_cfg.get('train_split_ratio', 0.8)
        random_seed  = loader_cfg.get('random_seed', 42)
    except Exception as e:
        print("Error extracting configuration:", e)
        return None, None

    # --- Load Label Map ---
    print(f"\nLoading label map from: {label_map_path}")
    if not os.path.exists(label_map_path):
        print("Error: Label map not found.")
        return None, None

    try:
        with open(label_map_path, 'r') as f:
            filename_label_map_str = json.load(f)
        print(f"Loaded map with {len(filename_label_map_str)} file entries.")
    except Exception as e:
        print("Error loading JSON map:", e)
        return None, None

    # --- Process Label Map ---
    print("Processing label map...")
    subject_id_label_map_numeric = {}
    subjects_with_errors = 0
    subjects_duplicate   = 0

    for filename, label_str in filename_label_map_str.items():
        subject_id = extract_id_from_filename(filename)
        if subject_id is None:
            subjects_with_errors += 1
            continue
        if isinstance(label_str, str) and "Error:" in label_str:
            subjects_with_errors += 1
            continue
        numeric_label = label_str_to_int.get(str(label_str).upper())
        if numeric_label is None:
            subjects_with_errors += 1
            continue
        if subject_id not in subject_id_label_map_numeric:
            subject_id_label_map_numeric[subject_id] = numeric_label
        else:
            subjects_duplicate += 1

    print(f"Created numeric map for {len(subject_id_label_map_numeric)} unique subject IDs.")
    if not subject_id_label_map_numeric:
        print("Error: No valid subjects found.")
        return None, None

    # --- Split Subject IDs into train and test sets ---
    all_subject_ids = list(subject_id_label_map_numeric.keys())
    random.shuffle(all_subject_ids)
    split_index     = int(len(all_subject_ids) * split_ratio)
    train_subject_ids = all_subject_ids[:split_index]
    test_subject_ids  = all_subject_ids[split_index:]
    print(f"\nSplitting data...\n  Training: {len(train_subject_ids)}, Testing: {len(test_subject_ids)}")

    if wandb_logger and wandb_logger.is_active:
        wandb_logger.log_data_summary({
            "data/total_subjects": len(all_subject_ids),
            "data/train_subjects": len(train_subject_ids),
            "data/test_subjects":  len(test_subject_ids)
        })

    # --- Create Datasets and DataLoaders ---
    print("\nSetting up Datasets and DataLoaders...")
    print("Creating Training Dataset...")
    train_dataset = FmriDataset(
        image_dir=fmri_dir,
        labels_dict=subject_id_label_map_numeric,
        target_shape=target_shape,
        subject_ids_to_use=train_subject_ids,
        id_extractor=extract_id_from_filename,
        apply_quantile_norm=apply_quantile_norm,
        apply_crop=apply_crop,
        crop_percentile=crop_percentile
    )
    print(f"Training dataset size: {len(train_dataset)}")
    if len(train_dataset) == 0:
        print("Error: Training dataset empty!")
        return None, None

    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_skip_error,
        pin_memory=True,
        drop_last=False
    )
    print(f"Training loader ready: {len(trainloader)} batches.")

    # --- Create Testing Dataset (if available) ---
    testloader = None
    test_dataset_size = 0
    if test_subject_ids:
        print("Creating Testing Dataset...")
        test_dataset = FmriDataset(
            image_dir=fmri_dir,
            labels_dict=subject_id_label_map_numeric,
            target_shape=target_shape,
            subject_ids_to_use=test_subject_ids,
            id_extractor=extract_id_from_filename,
            apply_quantile_norm=apply_quantile_norm,
            apply_crop=apply_crop,
            crop_percentile=crop_percentile
        )
        test_dataset_size = len(test_dataset)
        print(f"Testing dataset size: {test_dataset_size}")
        testloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn_skip_error,
            pin_memory=True,
            drop_last=False
        )
    else:
        print("No test subject IDs found, skipping test loader.")

    if wandb_logger and wandb_logger.is_active:
        wandb_logger.log_data_summary({
            "data/train_samples": len(train_dataset),
            "data/train_batches": len(trainloader),
            "data/test_samples":  test_dataset_size,
            "data/test_batches":  len(testloader) if testloader else 0
        })

    print("Dataset and DataLoader setup complete.")
    return trainloader, testloader
