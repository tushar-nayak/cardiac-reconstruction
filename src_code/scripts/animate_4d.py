import sys
import os
import argparse
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src_code.src.data_loaders.mitea_loader import MITEADataset
from src_code.src.models.ode_model import LatentEncoder, CardiacODE
from src_code.src.models.gaussian_model import GaussianModel

DEFAULT_DATA_DIR = "/home/sofa/host_dir/cardiac_reconstruction_project/cap-mitea/mitea"

def animate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir or os.path.join(os.path.dirname(args.checkpoint), "animations")
    os.makedirs(output_dir, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config', {})
    num_sparse_slices = int(config.get('num_sparse_slices', 3))
    encoder = LatentEncoder(input_channels=num_sparse_slices, latent_dim=args.latent_dim).to(device)
    ode = CardiacODE(latent_dim=args.latent_dim).to(device)
    gaussian_model = GaussianModel(num_gaussians=args.num_gaussians, latent_dim=args.latent_dim).to(device)

    print(f"Loading checkpoint from {args.checkpoint}...")
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    if 'ode_state_dict' not in checkpoint:
        raise ValueError("Checkpoint has no trained ODE; retrain with the current train.py")
    ode.load_state_dict(checkpoint['ode_state_dict'])
    gaussian_model.load_state_dict(checkpoint['gaussian_model_state_dict'])
    
    encoder.eval()
    ode.eval()
    gaussian_model.eval()

    dataset = MITEADataset(
        args.data_dir,
        split=args.split,
        num_sparse_slices=num_sparse_slices,
        split_seed=int(config.get('split_seed', 17)),
    )
    sample = dataset[args.sample_index]
    input_ed = sample['sparse_slices_ed'].unsqueeze(0).to(device)
    affine = sample['affine']
    D, H, W = sample['occupancy_ed'].shape

    t_steps = torch.linspace(0, 1, args.num_frames).to(device)
    
    with torch.no_grad():
        z_ed = encoder(input_ed)
        zt_seq = ode(z_ed, t_steps)
        z_slice = D // 2
        y_range = torch.linspace(0, H - 1, args.slice_resolution)
        x_range = torch.linspace(0, W - 1, args.slice_resolution)
        grid_y, grid_u = torch.meshgrid(y_range, x_range, indexing='ij')
        pts_img = torch.stack([grid_u, grid_y, torch.ones_like(grid_u) * z_slice], dim=-1).reshape(-1, 3)

        pts_h = torch.cat([pts_img, torch.ones(pts_img.shape[0], 1)], dim=-1)
        affine_th = affine.float() if torch.is_tensor(affine) else torch.from_numpy(affine).float()
        pts_world = (pts_h @ affine_th.T)[:, :3].to(device)
        
        frames = []
        print("Generating frames...")
        for i in range(args.num_frames):
            z_t = zt_seq[i]
            params_t = gaussian_model(z_t)
            occ_t = gaussian_model.evaluate_occupancy(pts_world.unsqueeze(0), params_t)
            frames.append(occ_t.view(args.slice_resolution, args.slice_resolution).cpu().numpy())

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(frames[0], cmap='magma', vmin=0, vmax=1)
    ax.set_title("4D Ventricular Reconstruction (Axial Slice)")
    ax.axis('off')

    def update(frame_idx):
        im.set_data(frames[frame_idx])
        progress = (frame_idx / max(args.num_frames - 1, 1)) * 100
        ax.set_title(f"Reconstruction Progress: {progress:.0f}% (ED to ES)")
        return [im]

    ani = FuncAnimation(fig, update, frames=args.num_frames, interval=200, blit=True)
    
    gif_path = os.path.join(output_dir, "ventricle_contraction.gif")
    ani.save(gif_path, writer=PillowWriter(fps=5))
    print(f"Animation saved to {gif_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-gaussians", type=int, default=5000)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--slice-resolution", type=int, default=100)
    animate(parser.parse_args())
