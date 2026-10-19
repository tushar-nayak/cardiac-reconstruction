import os
import glob
import random
import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset

class MITEADataset(Dataset):
    def __init__(self, data_dir, split='train', num_sparse_slices=3, transform=None, split_seed=17,
                 split_policy='shuffled'):
        """
        Dataset for MITEA data, simulating sparse 2D slice inputs for 4D reconstruction.
        """
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'images')
        self.labels_dir = os.path.join(data_dir, 'labels')
        if not 1 <= num_sparse_slices <= 3:
            raise ValueError("num_sparse_slices must be between 1 and 3")
        self.num_sparse_slices = num_sparse_slices
        self.transform = transform
        
        # Find subjects
        all_files = glob.glob(os.path.join(self.images_dir, 'MITEA_*_scan1_ED.nii.gz'))
        subject_ids = sorted([os.path.basename(f).split('_scan')[0] for f in all_files])
        
        if split not in {'train', 'val', 'test'}:
            raise ValueError("split must be one of: train, val, test")
        if not subject_ids:
            raise FileNotFoundError(f"No MITEA ED scans found under {self.images_dir}")

        if split_policy not in {'shuffled', 'may2026'}:
            raise ValueError("split_policy must be shuffled or may2026")
        self.split_policy = split_policy
        if split_policy == 'may2026':
            # Exact subject ordering and boundaries used by the May reference.
            self.subject_ids = {
                'train': subject_ids[:110], 'val': subject_ids[110:125],
                'test': subject_ids[125:],
            }[split]
            return

        # Shuffle once with a recorded seed so the split is reproducible and
        # does not depend on filesystem enumeration order.
        shuffled = list(subject_ids)
        random.Random(split_seed).shuffle(shuffled)
        n = len(shuffled)
        train_end = max(1, int(round(n * 0.8)))
        val_end = min(n, train_end + max(1, int(round(n * 0.1))))
        self.subject_ids = {
            'train': shuffled[:train_end],
            'val': shuffled[train_end:val_end],
            'test': shuffled[val_end:],
        }[split]

    def __len__(self):
        return len(self.subject_ids)

    def _normalize(self, data):
        data = data.astype(np.float32)
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val - min_val > 0:
            data = (data - min_val) / (max_val - min_val)
        return data

    def _pad_to_size(self, tensor, size=(200, 200)):
        c_h, c_w = tensor.shape
        
        if c_h > size[0] or c_w > size[1]:
            # Center crop if larger
            start_h = (c_h - size[0]) // 2 if c_h > size[0] else 0
            start_w = (c_w - size[1]) // 2 if c_w > size[1] else 0
            tensor = tensor[start_h:start_h+size[0], start_w:start_w+size[1]]
            c_h, c_w = tensor.shape

        pad_h = max(0, size[0] - c_h)
        pad_w = max(0, size[1] - c_w)
        # Pad with zeros
        return F.pad(tensor, (0, pad_w, 0, pad_h))

    @staticmethod
    def _to_canonical(data):
        """Convert nibabel ``(x, y, z)`` arrays to tensor ``(z, y, x)``."""
        return np.transpose(data, (2, 1, 0)).copy()

    def _get_slice_pose(self, affine, slice_idx, axis, shape):
        """
        Calculates the camera pose (R, T) for a given slice.
        axis: 0 (h), 1 (w), 2 (d)
        """
        # Centers of the volume in image coords
        d, h, w = shape
        center_img = torch.tensor([w / 2, h / 2, d / 2, 1.0])
        center_world = (torch.from_numpy(affine).float() @ center_img)[:3]
        
        # This is a simplification. For a real differentiable rasterizer,
        # we need the exact plane equations or viewing transformation.
        # For now, let's return the affine and slice metadata.
        return {
            'affine': torch.from_numpy(affine).float(),
            'slice_idx': slice_idx,
            'axis': axis,
            'center_world': center_world
        }

    def __getitem__(self, idx):
        subject = self.subject_ids[idx]
        
        # Load Scan 1 ED and ES
        img_ed_path = os.path.join(self.images_dir, f"{subject}_scan1_ED.nii.gz")
        img_es_path = os.path.join(self.images_dir, f"{subject}_scan1_ES.nii.gz")
        
        # Load labels for validation/supervision
        lbl_ed_path = os.path.join(self.labels_dir, f"{subject}_scan1_ED.nii.gz")
        lbl_es_path = os.path.join(self.labels_dir, f"{subject}_scan1_ES.nii.gz")
        
        img_ed_nii = nib.load(img_ed_path)
        img_es_nii = nib.load(img_es_path)
        lbl_ed_nii = nib.load(lbl_ed_path)
        lbl_es_nii = nib.load(lbl_es_path)
        
        img_ed = self._to_canonical(self._normalize(img_ed_nii.get_fdata()))
        img_es = self._to_canonical(self._normalize(img_es_nii.get_fdata()))
        
        lbl_ed = lbl_ed_nii.get_fdata().astype(np.float32)
        lbl_ed = (lbl_ed > 0).astype(np.float32)
        lbl_ed = self._to_canonical(lbl_ed) # (D, H, W)

        lbl_es = lbl_es_nii.get_fdata().astype(np.float32)
        lbl_es = (lbl_es > 0).astype(np.float32)
        lbl_es = self._to_canonical(lbl_es) # (D, H, W)
        
        # Extract middle slices
        d, h, w = img_ed.shape
        
        # Slices along different axes
        slices_ed = [
            torch.from_numpy(img_ed[d // 2, :, :]),
            torch.from_numpy(img_ed[:, h // 2, :]),
            torch.from_numpy(img_ed[:, :, w // 2]),
        ]
        
        # Metadata for each slice
        poses_ed = [
            self._get_slice_pose(img_ed_nii.affine, d // 2, 2, (d, h, w)),
            self._get_slice_pose(img_ed_nii.affine, h // 2, 1, (d, h, w)),
            self._get_slice_pose(img_ed_nii.affine, w // 2, 0, (d, h, w)),
        ]

        slices_es = [
            torch.from_numpy(img_es[d // 2, :, :]),
            torch.from_numpy(img_es[:, h // 2, :]),
            torch.from_numpy(img_es[:, :, w // 2]),
        ]
        
        # Pad slices
        slices_ed = torch.stack([self._pad_to_size(s) for s in slices_ed[:self.num_sparse_slices]])
        slices_es = torch.stack([self._pad_to_size(s) for s in slices_es[:self.num_sparse_slices]])

        return {
            'subject': subject,
            'index': idx,
            'sparse_slices_ed': slices_ed,
            'sparse_slices_es': slices_es,
            'poses_ed': poses_ed[:self.num_sparse_slices],
            'occupancy_ed': torch.from_numpy(lbl_ed).float(),
            'occupancy_es': torch.from_numpy(lbl_es).float(),
            'affine': torch.from_numpy(img_ed_nii.affine).float()
        }
