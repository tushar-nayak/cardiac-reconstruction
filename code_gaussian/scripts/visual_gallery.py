import sys
import os
import argparse
import torch
import matplotlib.pyplot as plt

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from code_gaussian.src.data_loaders.mitea_loader import MITEADataset
from code_gaussian.src.models.ode_model import LatentEncoder
from code_gaussian.src.models.gaussian_model import GaussianModel

def infer_num_gaussians(checkpoint, fallback):
    if checkpoint.get('type') == 'subject_gaussian_fit':
        return int(checkpoint['num_gaussians'])
    state = checkpoint.get('gaussian_model_state_dict', {})
    means = state.get('means')
    return int(means.shape[0]) if means is not None else fallback

def load_params(checkpoint, sample, device, latent_dim, num_gaussians):
    if checkpoint.get('type') == 'subject_gaussian_fit':
        return {
            'means': checkpoint['means'].to(device),
            'scales': checkpoint['scales'].to(device),
            'opacities': checkpoint['opacities'].to(device),
        }

    encoder = LatentEncoder(sample['sparse_slices_ed'].shape[0], latent_dim).to(device)
    gaussian_model = GaussianModel(num_gaussians, latent_dim).to(device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    gaussian_model.load_state_dict(checkpoint['gaussian_model_state_dict'])
    encoder.eval(); gaussian_model.eval()
    z_ed = encoder(sample['sparse_slices_ed'].unsqueeze(0).to(device))
    return gaussian_model(z_ed)

def generate_gallery(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = args.output_path or os.path.join(os.path.dirname(args.checkpoint), "visual_gallery.png")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    num_gaussians = infer_num_gaussians(checkpoint, args.num_gaussians)

    config = checkpoint.get('config', {})
    dataset = MITEADataset(
        args.data_dir,
        split=args.split,
        num_sparse_slices=int(config.get('num_sparse_slices', 3)),
        split_seed=int(config.get('split_seed', 17)),
        split_policy=config.get('split_policy', 'shuffled'),
    )
    sample = dataset[args.sample_index]
    if checkpoint.get('subject') and checkpoint['subject'] != sample['subject']:
        raise ValueError(f"Checkpoint belongs to {checkpoint['subject']}, selected sample is {sample['subject']}")
    gt_vol = sample['occupancy_ed'].cpu().numpy()
    affine = sample['affine']
    D, H, W = gt_vol.shape

    res = args.volume_resolution
    z = torch.linspace(0, D - 1, res)
    y = torch.linspace(0, H - 1, res)
    x = torch.linspace(0, W - 1, res)
    grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
    pts_img = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
    
    pts_h = torch.cat([pts_img, torch.ones(pts_img.shape[0], 1)], dim=-1)
    affine_th = affine.float() if torch.is_tensor(affine) else torch.from_numpy(affine).float()
    pts_world = (pts_h @ affine_th.T)[:, :3].to(device)

    with torch.no_grad():
        params_ed = load_params(checkpoint, sample, device, args.latent_dim, num_gaussians)
        model = GaussianModel(num_gaussians, args.latent_dim).to(device)
        occ_pred = model.evaluate_occupancy(pts_world.unsqueeze(0), params_ed)
        pred_vol = occ_pred.view(res, res, res).cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(gt_vol[D//2, :, :], cmap='gray'); axes[0, 0].set_title("GT Axial")
    axes[0, 1].imshow(gt_vol[:, H//2, :], cmap='gray'); axes[0, 1].set_title("GT Coronal")
    axes[0, 2].imshow(gt_vol[:, :, W//2], cmap='gray'); axes[0, 2].set_title("GT Sagittal")
    
    axes[1, 0].imshow(pred_vol[res//2, :, :], cmap='magma'); axes[1, 0].set_title("Pred Axial")
    axes[1, 1].imshow(pred_vol[:, res//2, :], cmap='magma'); axes[1, 1].set_title("Pred Coronal")
    axes[1, 2].imshow(pred_vol[:, :, res//2], cmap='magma'); axes[1, 2].set_title("Pred Sagittal")

    for ax in axes.ravel(): ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Gallery saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="runs/stabilization_v01/checkpoint_epoch_20.pth")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-gaussians", type=int, default=1000)
    parser.add_argument("--volume-resolution", type=int, default=64)
    generate_gallery(parser.parse_args())
