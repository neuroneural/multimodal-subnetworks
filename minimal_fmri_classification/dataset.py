import os
import re
import warnings
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import torch
from torch.utils.data import Dataset, DataLoader

# Helper function: Quantile Normalization
def qnormalize(img, qmin=0.02, qmax=0.98):
    """Applies quantile-based normalization to scale image intensities to [0, 1]."""
    qlow = torch.quantile(img, qmin)
    qhigh = torch.quantile(img, qmax)
    if qhigh - qlow > 1e-6:
        img = (img - qlow) / (qhigh - qlow)
    else:
        img = torch.zeros_like(img)
    return torch.clamp(img, 0, 1)

# Helper function: Tensor Cropping
def crop_tensor(tensor, percentile=10):
    """Crops tensor based on intensity percentile."""
    if tensor.numel() == 0: return tensor
    threshold = torch.quantile(tensor.float().flatten(), percentile / 100.0)
    mask = tensor > threshold
    if not torch.any(mask): return tensor
    nonzero = torch.nonzero(mask, as_tuple=False)
    if nonzero.numel() == 0: return tensor
    min_coords = torch.min(nonzero, dim=0)[0]
    max_coords = torch.max(nonzero, dim=0)[0]

    if len(min_coords) < 4: return tensor # Cannot perform 3D crop

    slices = [slice(None)] # Channel dim
    for dim_idx in range(1, tensor.dim()):
        if dim_idx < len(min_coords):
            min_c = int(min_coords[dim_idx].item())
            max_c = int(max_coords[dim_idx].item())
            if min_c < max_c + 1 and min_c < tensor.shape[dim_idx] and max_c < tensor.shape[dim_idx]:
                 slices.append(slice(min_c, max_c + 1))
            else:
                 slices.append(slice(None))
        else:
             slices.append(slice(None))
    return tensor[tuple(slices)]

# --- Main Dataset Class ---
class FmriDataset(Dataset):
    def __init__(self, image_dir, labels_dict, target_shape, subject_ids_to_use=None,
                 id_extractor=None, apply_quantile_norm=True, apply_crop=False,
                 crop_percentile=10):
        """
        Dataset for loading, preprocessing, and serving fMRI NIfTI files.
        Always uses the first volume for 4D scans.

        Args:
            image_dir (str): Directory containing NIfTI files (.nii or .nii.gz).
            labels_dict (dict): Mapping from subject_id (str) to numeric label.
            target_shape (tuple): Desired spatial shape (Depth, Height, Width).
            subject_ids_to_use (list, optional): List of subject IDs to include.
            id_extractor (callable, optional): Function to extract subject ID from filename.
            apply_quantile_norm (bool): Apply quantile normalization. Default True.
            apply_crop (bool): Apply intensity percentile cropping. Default False.
            crop_percentile (int or float): Percentile for cropping. Default 10.
        """
        self.image_dir = image_dir
        self.full_labels_dict = labels_dict
        self.target_shape = tuple(map(int, target_shape))
        self.id_extractor = id_extractor
        self.apply_quantile_norm = apply_quantile_norm
        self.apply_crop = apply_crop
        self.crop_percentile = crop_percentile

        self.valid_files = []
        self.file_to_label = {}

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not callable(self.id_extractor):
            warnings.warn("No valid id_extractor function provided.", UserWarning)

        print(f"Scanning NIfTI files in: {self.image_dir}")
        try:
            all_nifti_files = sorted([
                f for f in os.listdir(self.image_dir)
                if f.endswith(('.nii', '.nii.gz')) and not f.startswith('.')
            ])
        except OSError as e:
            raise OSError(f"Could not read directory {self.image_dir}: {e}")

        ids_to_include = set(subject_ids_to_use) if subject_ids_to_use is not None else set(self.full_labels_dict.keys())
        files_processed = 0

        for filename in all_nifti_files:
            subject_id = self.id_extractor(filename) if callable(self.id_extractor) else None
            if subject_id is None: continue
            if subject_id not in self.full_labels_dict: continue
            if subject_id not in ids_to_include: continue

            self.valid_files.append(filename)
            self.file_to_label[filename] = self.full_labels_dict[subject_id]
            files_processed += 1

        print(f"Scan complete. Found {files_processed} valid files.")
        if not self.valid_files:
            warnings.warn("Dataset initialization finished, but no valid files were found.", UserWarning)

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        if not 0 <= idx < len(self.valid_files):
            raise IndexError(f"Index {idx} out of range.")

        filename = self.valid_files[idx]
        filepath = os.path.join(self.image_dir, filename)
        label = self.file_to_label[filename]

        try:
            img_nii = nib.load(filepath)
            img_data_full = img_nii.get_fdata(dtype=np.float32, caching='unchanged')
        except Exception as e:
            warnings.warn(f"Error loading NIfTI file {filepath}: {e}. Returning None.", UserWarning)
            return None

        if img_data_full.ndim == 4:
            if img_data_full.shape[3] > 0:
                 img_data_3d = img_data_full[..., 0]
            else: return None # 4th dim empty
        elif img_data_full.ndim == 3:
            img_data_3d = img_data_full
        else: return None # Unexpected dimensions

        if img_data_3d.shape != self.target_shape:
            current_shape = img_data_3d.shape
            zoom_factors = [
                target_dim / current_dim if current_dim > 0 else 0
                for target_dim, current_dim in zip(self.target_shape, current_shape)
            ]
            if any(zf <= 0 for zf in zoom_factors):
                 pass # Skip resampling if factor is invalid, shape mismatch will occur
            else:
                try:
                    img_data_3d_resampled = zoom(img_data_3d, zoom_factors, order=1, mode='nearest', prefilter=False)
                    if img_data_3d_resampled.shape != self.target_shape:
                         adjusted_data = np.zeros(self.target_shape, dtype=np.float32)
                         crop_slice = tuple(slice(0, min(t, s)) for t, s in zip(self.target_shape, img_data_3d_resampled.shape))
                         paste_slice = tuple(slice(0, min(t, s)) for t, s in zip(self.target_shape, img_data_3d_resampled.shape))
                         adjusted_data[crop_slice] = img_data_3d_resampled[paste_slice]
                         img_data_3d = adjusted_data
                    else:
                         img_data_3d = img_data_3d_resampled
                except Exception as e:
                     warnings.warn(f"Error during resampling for {filename}: {e}. Using original data.", UserWarning)

        img_data_3d = img_data_3d.astype(np.float32)
        img_tensor = torch.from_numpy(img_data_3d).unsqueeze(0).float()

        if self.apply_quantile_norm:
            img_tensor = qnormalize(img_tensor, qmin=0.02, qmax=0.98)
        if self.apply_crop:
            img_tensor = crop_tensor(img_tensor, percentile=self.crop_percentile)

        label_tensor = torch.tensor(label, dtype=torch.float32)
        return img_tensor, label_tensor

# --- Collate Function ---
def collate_fn_skip_error(batch):
    """Collate function that filters out None items before batching."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([]) # Return empty tensors if batch is empty
    try:
        return torch.utils.data.dataloader.default_collate(batch)
    except RuntimeError as e:
        print(f"Error during collate: {e}. Skipping batch.")
        # for i, (img, lab) in enumerate(batch): print(f" Item {i} shape: {img.shape}")
        return torch.tensor([]), torch.tensor([])