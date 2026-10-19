import sys
import os
import torch
import numpy as np

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from code_gaussian.src.data_loaders.mitea_loader import MITEADataset
from code_gaussian.src.utils.coord_utils import sample_volume_at_points

def check_labels():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "/home/sofa/host_dir/cardiac_reconstruction_project/cap-mitea/mitea"
    dataset = MITEADataset(data_dir, split='val')
    sample = dataset[0]
    
    vol = sample['occupancy_ed']
    affine = sample['affine']
    
    # Sample points specifically near occupied regions
    occ_indices = torch.nonzero(vol)
    print(f"Total occupied voxels: {len(occ_indices)}")
    
    selected_indices = occ_indices[:10]
    pts_img = selected_indices[:, [2, 1, 0]].float() # (W, H, D)
    pts_h = torch.cat([pts_img, torch.ones(len(pts_img), 1)], dim=-1)
    pts_world = (pts_h @ affine.T)[:, :3]
    
    target_occ = sample_volume_at_points([vol], pts_world.unsqueeze(0), affine.unsqueeze(0))
    print(f"Target occupancy for known occupied points: {target_occ}")

if __name__ == "__main__":
    check_labels()
