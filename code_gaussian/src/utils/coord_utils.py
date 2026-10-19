import torch
import torch.nn.functional as F

def sample_volume_at_points(volumes, query_points, affines):
    """
    Samples volumes at given query points.
    
    volumes: List of Tensors of shape (D, H, W)
    query_points: (B, N, 3) in world coordinates
    affines: (B, 4, 4) affine matrices (Image to World)
    
    Returns: (B, N) sampled values
    """
    B, N, _ = query_points.shape
    device = query_points.device
    sampled_values = []

    for i in range(B):
        vol = volumes[i].to(device) # (D, H, W)
        affine = affines[i].to(device) # (4, 4)
        points = query_points[i] # (N, 3)
        
        # Invert affine: World to Image
        inv_affine = torch.inverse(affine)
        
        # Add homogeneous coordinate
        points_h = torch.cat([points, torch.ones(N, 1).to(device)], dim=-1) # (N, 4)
        
        # Transform to image coordinates
        # (N, 4) @ (4, 4).T -> (N, 4)
        points_img = (points_h @ inv_affine.T)[:, :3]
        
        # Convert to grid coordinates [-1, 1]
        # volumes[i] is (D, H, W)
        # grid_sample expects (1, C, D, H, W) and grid (1, D_out, H_out, W_out, 3)
        # Here we want to sample N points, so we can use grid of (1, 1, 1, N, 3)
        
        size = torch.tensor([vol.shape[2], vol.shape[1], vol.shape[0]]).to(device) # (W, H, D)
        grid_points = (2.0 * points_img / (size - 1.0)) - 1.0
        
        # grid_sample expects (B, C, D, H, W) and grid (B, D_out, H_out, W_out, 3)
        # and grid values are (x, y, z) where x is for last dim (W), y for H, z for D.
        # Our points_img is (x_idx, y_idx, z_idx) where x_idx is for W, y_idx for H, z_idx for D.
        # Wait, nibabel/nilearn usually uses (W, H, D) or (H, W, D)?
        # MITEADataset: img_ed = self._normalize(img_ed_nii.get_fdata())
        # fdata is typically (W, H, D) for Nifti if read by nibabel.
        # Let's assume it matches.
        
        grid = grid_points.view(1, 1, 1, N, 3)
        v = vol.view(1, 1, vol.shape[0], vol.shape[1], vol.shape[2]) # (1, 1, D, H, W)
        
        # grid_sample input: (N, C, D, H, W), grid: (N, D_out, H_out, W_out, 3)
        # grid values: (x, y, z)
        # x refers to the size[2] (W), y to size[1] (H), z to size[0] (D)
        
        out = F.grid_sample(v, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled_values.append(torch.clamp(out.view(N), 0, 1))

    return torch.stack(sampled_values)
