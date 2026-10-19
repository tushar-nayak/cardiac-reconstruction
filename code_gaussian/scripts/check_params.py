import torch
import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from code_gaussian.src.models.gaussian_model import GaussianModel

def check_params():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "runs/smoke_run_01/checkpoint_epoch_20.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    gaussian_state = checkpoint['gaussian_model_state_dict']
    
    means = gaussian_state['means']
    scales = gaussian_state['scales']
    opacities = gaussian_state['opacities']
    
    print(f"Means - min: {means.min().item():.4f}, max: {means.max().item():.4f}, mean: {means.mean().item():.4f}")
    print(f"Scales - min: {scales.min().item():.4f}, max: {scales.max().item():.4f}, mean: {scales.mean().item():.4f}")
    print(f"Opacities - min: {opacities.min().item():.4f}, max: {opacities.max().item():.4f}, mean: {opacities.mean().item():.4f}")
    
    # Check deformation net outputs for a dummy latent
    latent = torch.randn(1, 128).to(device)
    model = GaussianModel(num_gaussians=5000, latent_dim=128).to(device)
    model.load_state_dict(gaussian_state)
    
    params = model(latent)
    print(f"\nAfter forward pass with random latent:")
    print(f"Current Means - min: {params['means'].min().item():.4f}, max: {params['means'].max().item():.4f}")
    print(f"Current Scales - min: {params['scales'].min().item():.4f}, max: {params['scales'].max().item():.4f}")
    print(f"Current Opacities - min: {params['opacities'].min().item():.4f}, max: {params['opacities'].max().item():.4f}")

if __name__ == "__main__":
    check_params()
