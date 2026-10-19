import torch
import torch.nn as nn

class RadiologicalRasterizer(nn.Module):
    def __init__(self):
        super(RadiologicalRasterizer, self).__init__()

    def forward(self, gaussian_params, pose, slice_shape=(200, 200), point_chunk=2048, gaussian_chunk=256):
        """
        Differentiable rasterizer with chunking to limit memory use.
        """
        device = gaussian_params['means'].device
        B = gaussian_params['means'].shape[0]
        H, W = slice_shape
        means = gaussian_params['means']
        scales = gaussian_params['scales']
        opacities = gaussian_params['opacities']
        intensities = gaussian_params['intensities']
        if intensities.dim() == 2:
            intensities = intensities.unsqueeze(0).expand(B, -1, -1)

        u = torch.linspace(0, W - 1, W, device=device)
        v = torch.linspace(0, H - 1, H, device=device)
        grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')

        pts_img = torch.zeros(H, W, 3, device=device)
        axis = pose['axis']
        slice_idx = pose['slice_idx']

        if axis == 0:
            pts_img[..., 0] = slice_idx
            pts_img[..., 1] = grid_u
            pts_img[..., 2] = grid_v
        elif axis == 1:
            pts_img[..., 0] = grid_u
            pts_img[..., 1] = slice_idx
            pts_img[..., 2] = grid_v
        else:
            pts_img[..., 0] = grid_u
            pts_img[..., 1] = grid_v
            pts_img[..., 2] = slice_idx

        pts_h = torch.cat([pts_img, torch.ones(H, W, 1, device=device)], dim=-1)
        pts_world = (pts_h @ pose['affine'].to(device).T)[..., :3]

        if 'R_delta' in pose:
            center = pose['center_world'].to(device)
            pts_world = (pts_world - center) @ pose['R_delta'].to(device).T + center + pose['T_delta'].to(device)

        pts_flat = pts_world.view(-1, 3)
        final_signal = torch.zeros(B, pts_flat.shape[0], device=device)
        oi = opacities * intensities
        num_gaussians = means.shape[1]

        for p_start in range(0, pts_flat.shape[0], point_chunk):
            p_chunk = pts_flat[p_start:p_start + point_chunk]
            chunk_signal = torch.zeros(B, p_chunk.shape[0], device=device)

            for g_start in range(0, num_gaussians, gaussian_chunk):
                g_end = g_start + gaussian_chunk
                means_chunk = means[:, g_start:g_end]
                scales_chunk = scales[:, g_start:g_end]
                oi_chunk = oi[:, g_start:g_end]

                diff = p_chunk.unsqueeze(0).unsqueeze(2) - means_chunk.unsqueeze(1)
                dist_sq = torch.sum((diff / (scales_chunk.unsqueeze(1) + 1e-8)) ** 2, dim=-1)
                weights = torch.exp(-0.5 * dist_sq)
                chunk_signal = chunk_signal + torch.sum(weights * oi_chunk.transpose(-1, -2), dim=-1)

            final_signal[:, p_start:p_start + p_chunk.shape[0]] = chunk_signal

        return final_signal.view(B, H, W)
