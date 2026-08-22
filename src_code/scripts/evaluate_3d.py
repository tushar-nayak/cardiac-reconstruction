import sys
import os
import argparse
import json
import torch
import matplotlib.pyplot as plt

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src_code.src.data_loaders.mitea_loader import MITEADataset
from src_code.src.models.ode_model import LatentEncoder
from src_code.src.models.gaussian_model import GaussianModel
from src_code.src.utils.coord_utils import sample_volume_at_points
from src_code.src.metrics import binary_metrics, save_mesh_npz, volume_to_mesh

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

    input_ed = sample['sparse_slices_ed'].unsqueeze(0).to(device)
    z_ed = encoder(input_ed)
    return gaussian_model(z_ed)

def image_to_world(points_img, affine, device):
    points_h = torch.cat([points_img, torch.ones(points_img.shape[0], 1, device=device)], dim=-1)
    return (points_h @ affine.to(device).T)[:, :3]

def sample_metric_points(gt_vol, affine, num_points, device):
    volume = torch.from_numpy(gt_vol).to(device)
    occ_indices = torch.nonzero(volume)
    if len(occ_indices) == 0:
        raise ValueError("Selected sample has no occupied voxels.")

    num_pos = num_points // 3
    num_centroid = num_points // 3
    num_uniform = num_points - num_pos - num_centroid

    sel = occ_indices[torch.randint(0, len(occ_indices), (num_pos,), device=device)]
    pts_pos_img = sel[:, [2, 1, 0]].float()

    centroid_img = occ_indices.float().mean(dim=0)[[2, 1, 0]]
    spread_img = occ_indices.float().std(dim=0, unbiased=False)[[2, 1, 0]].clamp_min(4.0)
    pts_centroid_img = centroid_img.unsqueeze(0) + torch.randn(num_centroid, 3, device=device) * (spread_img * 0.15)

    D, H, W = volume.shape
    pts_uniform_img = torch.stack([
        torch.rand(num_uniform, device=device) * (W - 1),
        torch.rand(num_uniform, device=device) * (H - 1),
        torch.rand(num_uniform, device=device) * (D - 1),
    ], dim=-1)

    points_img = torch.cat([pts_pos_img, pts_centroid_img, pts_uniform_img], dim=0)
    regions = {
        'wall': slice(0, num_pos),
        'centroid': slice(num_pos, num_pos + num_centroid),
        'uniform': slice(num_pos + num_centroid, num_points),
    }
    return image_to_world(points_img, affine, device), regions

def evaluate_3d(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir or os.path.join(os.path.dirname(args.checkpoint), "eval_3d")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading checkpoint {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    num_gaussians = infer_num_gaussians(checkpoint, args.num_gaussians)

    config = checkpoint.get('config', {})
    dataset = MITEADataset(
        args.data_dir,
        split=args.split,
        num_sparse_slices=int(config.get('num_sparse_slices', 3)),
        split_seed=int(config.get('split_seed', 17)),
    )
    sample = dataset[args.sample_index]
    gt_vol = sample['occupancy_ed'].cpu().numpy() # (D, H, W)
    affine = sample['affine']
    D, H, W = gt_vol.shape

    test_idx = torch.tensor([D//2, H//2, W//2]).float()
    test_pts_img = test_idx[[2, 1, 0]].view(1, 3)
    pts_h = torch.cat([test_pts_img, torch.ones(1, 1)], dim=-1)
    pts_world = (pts_h @ affine.T)[:, :3]
    
    inv_affine = torch.inverse(affine)
    pts_h_world = torch.cat([pts_world, torch.ones(1, 1)], dim=-1)
    back_img = (pts_h_world @ inv_affine.T)[:, :3]
    print(f"Coord Test - Original: {test_pts_img.tolist()}, Recovered: {back_img.tolist()}")

    z_idx = D // 2
    u = torch.linspace(0, W - 1, args.slice_resolution)
    v = torch.linspace(0, H - 1, args.slice_resolution)
    grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')
    pts_img_axial = torch.stack([grid_u, grid_v, torch.ones_like(grid_u) * z_idx], dim=-1).reshape(-1, 3)
    
    y_idx = H // 2
    u = torch.linspace(0, W - 1, args.slice_resolution)
    w = torch.linspace(0, D - 1, args.slice_resolution)
    grid_w, grid_u = torch.meshgrid(w, u, indexing='ij')
    pts_img_coronal = torch.stack([grid_u, torch.ones_like(grid_u) * y_idx, grid_w], dim=-1).reshape(-1, 3)

    with torch.no_grad():
        params_ed = load_params(checkpoint, sample, device, args.latent_dim, num_gaussians)
        
        pts_axial_world = (torch.cat([pts_img_axial, torch.ones(pts_img_axial.shape[0], 1)], dim=-1) @ affine.T)[:, :3].to(device)
        model = GaussianModel(num_gaussians, args.latent_dim).to(device)
        occ_axial = model.evaluate_occupancy(pts_axial_world.unsqueeze(0), params_ed).view(args.slice_resolution, args.slice_resolution).cpu().numpy()
        
        pts_coronal_world = (torch.cat([pts_img_coronal, torch.ones(pts_img_coronal.shape[0], 1)], dim=-1) @ affine.T)[:, :3].to(device)
        occ_coronal = model.evaluate_occupancy(pts_coronal_world.unsqueeze(0), params_ed).view(args.slice_resolution, args.slice_resolution).cpu().numpy()

        metric_points, regions = sample_metric_points(gt_vol, affine, args.metric_points, device)
        metric_pred = model.evaluate_occupancy(metric_points.unsqueeze(0), params_ed)
        metric_target = sample_volume_at_points([sample['occupancy_ed']], metric_points.unsqueeze(0), affine.unsqueeze(0).to(device))

        # Evaluate the entire canonical volume on a common grid. This is less
        # optimistic than the old hand-picked point sample and is also the
        # source used for reproducible mesh extraction.
        res = args.volume_resolution
        z = torch.linspace(0, D - 1, res, device=device)
        y = torch.linspace(0, H - 1, res, device=device)
        x = torch.linspace(0, W - 1, res, device=device)
        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
        grid_img = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
        grid_world = image_to_world(grid_img, affine, device)
        pred_volume = model.evaluate_occupancy(grid_world.unsqueeze(0), params_ed).reshape(res, res, res)
        target_volume = sample_volume_at_points(
            [sample['occupancy_ed']], grid_world.unsqueeze(0), affine.unsqueeze(0).to(device)
        ).reshape(res, res, res)

    pred_mask = metric_pred > args.threshold
    target_mask = metric_target > 0.5
    intersection = torch.logical_and(pred_mask, target_mask).sum().item()
    union = torch.logical_or(pred_mask, target_mask).sum().item()
    metrics = {
        'checkpoint': args.checkpoint,
        'sample_index': args.sample_index,
        'num_gaussians': num_gaussians,
        'threshold': args.threshold,
        'accuracy': torch.eq(pred_mask, target_mask).float().mean().item(),
        'iou': intersection / max(union, 1),
        'mean_pred_wall': metric_pred[:, regions['wall']].mean().item(),
        'mean_pred_centroid': metric_pred[:, regions['centroid']].mean().item(),
        'mean_pred_uniform': metric_pred[:, regions['uniform']].mean().item(),
        'mean_target_wall': metric_target[:, regions['wall']].mean().item(),
        'mean_target_centroid': metric_target[:, regions['centroid']].mean().item(),
        'wall_centroid_gap': (
            metric_pred[:, regions['wall']].mean() - metric_pred[:, regions['centroid']].mean()
        ).item(),
    }
    metrics['full_volume'] = binary_metrics(
        pred_volume.cpu().numpy(), target_volume.cpu().numpy(), args.threshold
    )

    scale_x = (W - 1) / max(args.volume_resolution - 1, 1)
    scale_y = (H - 1) / max(args.volume_resolution - 1, 1)
    scale_z = (D - 1) / max(args.volume_resolution - 1, 1)
    grid_affine = affine.cpu().numpy() @ torch.diag(torch.tensor(
        [scale_x, scale_y, scale_z, 1.0], dtype=torch.float32
    )).numpy()
    pred_vertices, pred_faces = volume_to_mesh(pred_volume.cpu().numpy(), grid_affine, args.threshold)
    target_vertices, target_faces = volume_to_mesh(target_volume.cpu().numpy(), grid_affine, 0.5)
    save_mesh_npz(os.path.join(output_dir, 'predicted_mesh.npz'), pred_vertices, pred_faces)
    save_mesh_npz(os.path.join(output_dir, 'target_mesh.npz'), target_vertices, target_faces)
    metrics['predicted_mesh'] = {'vertices': len(pred_vertices), 'faces': len(pred_faces)}
    metrics['target_mesh'] = {'vertices': len(target_vertices), 'faces': len(target_faces)}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(gt_vol[D//2, :, :], cmap='gray')
    axes[0, 0].set_title("GT Axial")
    axes[0, 1].imshow(occ_axial, cmap='magma', vmin=0, vmax=1)
    axes[0, 1].set_title("Pred Axial (Stabilized)")
    
    axes[1, 0].imshow(gt_vol[:, H//2, :], cmap='gray')
    axes[1, 0].set_title("GT Coronal")
    axes[1, 1].imshow(occ_coronal, cmap='magma', vmin=0, vmax=1)
    axes[1, 1].set_title("Pred Coronal (Stabilized)")

    for ax in axes.ravel(): ax.axis('off')
    
    save_path = os.path.join(output_dir, f"comparison_v01.png")
    plt.savefig(save_path)
    print(f"Evaluation saved to {save_path}")

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-gaussians", type=int, default=1000)
    parser.add_argument("--slice-resolution", type=int, default=128)
    parser.add_argument("--metric-points", type=int, default=3000)
    parser.add_argument("--volume-resolution", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    evaluate_3d(parser.parse_args())
