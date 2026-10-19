import torch
import nibabel as nib
import numpy as np
import os
import sys

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from code_gaussian.src.data_loaders.mitea_loader import MITEADataset

def debug_coords():
    data_dir = "/home/sofa/host_dir/cardiac_reconstruction_project/cap-mitea/mitea"
    dataset = MITEADataset(data_dir, split='val')
    sample = dataset[0]
    
    vol = sample['occupancy_ed']
    affine = sample['affine']
    
    # 1. Take a voxel index and convert to world
    occ_indices = torch.nonzero(vol)
    idx = occ_indices[0] # (D, H, W)
    print(f"Voxel index (D, H, W): {idx}")
    
    # In MITEADataset we do: pts_img = selected_indices[:, [2, 1, 0]] which is (W, H, D)
    pts_img = idx[[2, 1, 0]].float()
    print(f"Image coordinates (W, H, D): {pts_img}")
    
    pts_h = torch.cat([pts_img, torch.tensor([1.0])])
    pts_world = (pts_h @ affine.T)[:3]
    print(f"World coordinates: {pts_world}")
    
    # 2. Check a few landmarks if possible, or just the range
    print(f"Volume shape: {vol.shape}")
    corners_img = torch.tensor([
        [0, 0, 0, 1],
        [vol.shape[2]-1, 0, 0, 1],
        [0, vol.shape[1]-1, 0, 1],
        [0, 0, vol.shape[0]-1, 1],
        [vol.shape[2]-1, vol.shape[1]-1, vol.shape[0]-1, 1]
    ]).float()
    corners_world = (corners_img @ affine.T)[:, :3]
    print(f"Corners in world coords:\n{corners_world}")

if __name__ == "__main__":
    debug_coords()
