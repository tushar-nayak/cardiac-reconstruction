import sys
import os
import torch
import numpy as np
import argparse

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from code_gaussian.src.data_loaders.mitea_loader import MITEADataset
from code_gaussian.src.models.ode_model import LatentEncoder, CardiacODE
from code_gaussian.src.models.gaussian_model import GaussianModel

def check_stats(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "/home/sofa/host_dir/cardiac_reconstruction_project/cap-mitea/mitea"

    # Load models
    latent_dim = 128
    num_gaussians = 5000
    encoder = LatentEncoder(input_channels=3, latent_dim=latent_dim).to(device)
    ode = CardiacODE(latent_dim=latent_dim).to(device)
    gaussian_model = GaussianModel(num_gaussians=num_gaussians, latent_dim=latent_dim).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    ode.load_state_dict(checkpoint['ode_state_dict'])
    gaussian_model.load_state_dict(checkpoint['gaussian_model_state_dict'])
    
    encoder.eval()
    ode.eval()
    gaussian_model.eval()

    dataset = MITEADataset(data_dir, split='val')
    sample = dataset[0]
    input_ed = sample['sparse_slices_ed'].unsqueeze(0).to(device)
    gt_occ_ed = sample['occupancy_ed']
    affine = sample['affine']

    with torch.no_grad():
        z_ed = encoder(input_ed)
        params_ed = gaussian_model(z_ed)
        
        # Sample points specifically near GT occupied voxels
        occ_indices = torch.nonzero(gt_occ_ed)
        selected_indices = occ_indices[torch.randint(0, len(occ_indices), (1000,))]
        pts_img = selected_indices[:, [2, 1, 0]].float() # (W, H, D)
        
        pts_h = torch.cat([pts_img, torch.ones(pts_img.shape[0], 1)], dim=-1)
        affine_th = affine.float() if torch.is_tensor(affine) else torch.from_numpy(affine).float()
        pts_world = (pts_h @ affine_th.T)[:, :3].to(device)
        
        occ_pred = gaussian_model.evaluate_occupancy(pts_world.unsqueeze(0), params_ed)
        
        print(f"Stats for Predicted Occupancy at GT points:")
        print(f"Min: {occ_pred.min().item():.4f}")
        print(f"Max: {occ_pred.max().item():.4f}")
        print(f"Mean: {occ_pred.mean().item():.4f}")
        print(f"Values > 0.5: {(occ_pred > 0.5).sum().item()} / {occ_pred.numel()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    check_stats(args.checkpoint)
