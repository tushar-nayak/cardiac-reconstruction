import sys
import os

# Add the project root to sys.path to allow imports from src_code
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src_code.src.data_loaders.mitea_loader import MITEADataset
from src_code.src.models.ode_model import LatentEncoder
from src_code.src.models.gaussian_model import GaussianModel
from src_code.src.models.pose_model import PoseOptimizer
from src_code.src.utils.coord_utils import sample_volume_at_points
from src_code.src.rendering.rasterizer import RadiologicalRasterizer
import tqdm
import argparse

def collate_fn(batch):
    return {
        'subject': [d['subject'] for d in batch],
        'index': torch.tensor([d['index'] for d in batch]),
        'sparse_slices_ed': torch.stack([d['sparse_slices_ed'] for d in batch]),
        'poses_ed': [d['poses_ed'] for d in batch],
        'occupancy_ed': [d['occupancy_ed'] for d in batch],
        'affine': torch.stack([d['affine'] for d in batch])
    }

def sample_query_points(volume, affine, num_samples, device):
    occ_indices = torch.nonzero(volume)
    num_pos = num_samples // 2
    num_cavity = num_samples // 4
    num_uniform = num_samples - num_pos - num_cavity

    if len(occ_indices) == 0:
        points = torch.randn(num_samples, 3, device=device) * 100.0
        weights = torch.ones(num_samples, device=device)
        return points, weights

    sel = occ_indices[torch.randint(0, len(occ_indices), (num_pos,))]
    pts_pos_img = sel[:, [2, 1, 0]].float().to(device)
    pts_pos_img = pts_pos_img + torch.randn_like(pts_pos_img) * 2.0
    pts_pos_h = torch.cat([pts_pos_img, torch.ones(num_pos, 1, device=device)], dim=-1)
    pts_pos = (pts_pos_h @ affine.to(device).T)[:, :3]

    centroid_img = occ_indices.float().mean(dim=0)[[2, 1, 0]].to(device)
    spread_img = occ_indices.float().std(dim=0, unbiased=False)[[2, 1, 0]].to(device).clamp_min(4.0)
    cavity_noise = torch.randn(num_cavity, 3, device=device) * (spread_img * 0.15)
    pts_cavity_img = centroid_img.unsqueeze(0) + cavity_noise
    pts_cavity_h = torch.cat([pts_cavity_img, torch.ones(num_cavity, 1, device=device)], dim=-1)
    pts_cavity = (pts_cavity_h @ affine.to(device).T)[:, :3]

    pts_uniform = (torch.rand(num_uniform, 3, device=device) * 400.0) - 200.0

    points = torch.cat([pts_pos, pts_cavity, pts_uniform], dim=0)
    weights = torch.cat([
        torch.ones(num_pos, device=device),
        torch.full((num_cavity,), 25.0, device=device),
        torch.full((num_uniform,), 5.0, device=device),
    ])
    return points, weights

def train(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.run_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MITEADataset(
        args.data_dir,
        split=args.split,
        num_sparse_slices=args.num_sparse_slices,
        split_seed=args.split_seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    encoder = LatentEncoder(input_channels=args.num_sparse_slices, latent_dim=args.latent_dim).to(device)
    gaussian_model = GaussianModel(num_gaussians=args.num_gaussians, latent_dim=args.latent_dim).to(device)
    rasterizer = RadiologicalRasterizer().to(device)
    pose_optimizer = PoseOptimizer(num_subjects=len(dataset)).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(gaussian_model.parameters()) + list(pose_optimizer.parameters()),
        lr=args.lr
    )

    for epoch in range(args.epochs):
        pbar = tqdm.tqdm(enumerate(dataloader), desc=f"Epoch {epoch}", total=len(dataloader))
        for batch_idx, batch in pbar:
            if epoch == 0 and batch_idx == 0:
                gaussian_model.initialize_from_voxels(batch['occupancy_ed'], batch['affine'])
            
            input_ed = batch['sparse_slices_ed'].to(device)
            subject_indices = batch['index'].to(device)
            B = input_ed.shape[0]
            
            z_ed = encoder(input_ed)
            params_ed = gaussian_model(z_ed)
            refined_poses_ed = pose_optimizer(subject_indices, batch['poses_ed'])

            query_points = []
            query_weights = []
            for i in range(B):
                pts_i, weights_i = sample_query_points(
                    batch['occupancy_ed'][i],
                    batch['affine'][i],
                    args.num_samples,
                    device,
                )
                query_points.append(pts_i)
                query_weights.append(weights_i)

            query_points = torch.stack(query_points)
            query_weights = torch.stack(query_weights)

            occ_pred_ed = gaussian_model.evaluate_occupancy(query_points, params_ed)
            target_occ_ed = sample_volume_at_points(batch['occupancy_ed'], query_points, batch['affine'].to(device))

            weights = torch.where(
                target_occ_ed > 0.5,
                query_weights,
                query_weights * 4.0,
            )
            loss_occ = F.binary_cross_entropy(occ_pred_ed, target_occ_ed, weight=weights)

            loss_img = torch.tensor(0.0).to(device)
            for b in range(B):
                for s in range(len(refined_poses_ed[b])):
                    pose = refined_poses_ed[b][s]
                    params_b = {k: v[b:b+1] if v.dim() > 2 else v for k, v in params_ed.items()}
                    img_pred = rasterizer(params_b, pose)
                    img_target = batch['sparse_slices_ed'][b, s].to(device).unsqueeze(0)
                    loss_img += F.mse_loss(img_pred, img_target)
            
            loss_img = loss_img / (B * len(refined_poses_ed[0]))
            loss_sparse = torch.mean(params_ed['opacities']) + 0.01 * torch.mean(params_ed['scales'])
            loss = loss_occ + args.image_loss_weight * loss_img + args.sparse_loss_weight * loss_sparse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'occ': f"{loss_occ.item():.4f}", 'img': f"{loss_img.item():.4f}"})

        if (epoch + 1) % args.checkpoint_every == 0:
            torch.save({
                'format_version': 2,
                'model_type': 'encoder_gaussian_occupancy',
                'epoch': epoch,
                'config': vars(args),
                'encoder_state_dict': encoder.state_dict(),
                'gaussian_model_state_dict': gaussian_model.state_dict(),
                'pose_optimizer_state_dict': pose_optimizer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.run_dir, f"checkpoint_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--run-dir", type=str, default="runs/stabilization_v01")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-gaussians", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--num-sparse-slices", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--image-loss-weight", type=float, default=1.0)
    parser.add_argument("--sparse-loss-weight", type=float, default=0.1)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()
    train(args)
