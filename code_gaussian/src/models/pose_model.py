import torch
import torch.nn as nn


def axis_angle_to_matrix(axis_angle):
    """Convert axis-angle vectors to rotation matrices without PyTorch3D."""
    theta = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / theta.clamp_min(1e-8)
    x, y, z = axis.unbind(-1)
    zeros = torch.zeros_like(x)
    K = torch.stack((zeros, -z, y, z, zeros, -x, -y, x, zeros), dim=-1).reshape(*axis.shape[:-1], 3, 3)
    eye = torch.eye(3, device=axis.device, dtype=axis.dtype).expand_as(K)
    # Stable Rodrigues formula; the small-angle limit is identity.
    return eye + torch.sin(theta)[..., None] * K + (1 - torch.cos(theta))[..., None] * (K @ K)

class PoseOptimizer(nn.Module):
    def __init__(self, num_subjects, num_slices=3):
        """
        Learns small delta rotations and translations for each slice.
        """
        super(PoseOptimizer, self).__init__()
        # Initializing with small noise
        self.delta_r = nn.Parameter(torch.randn(num_subjects, num_slices, 3) * 1e-4)
        self.delta_t = nn.Parameter(torch.randn(num_subjects, num_slices, 3) * 1e-4)

    def forward(self, subject_indices, poses_batch):
        """
        Applies learned deltas to the input poses.
        subject_indices: (B)
        poses_batch: List of List of dicts (from MITEADataset)
        """
        B = len(poses_batch)
        S = len(poses_batch[0])
        device = self.delta_r.device
        
        refined_poses = []
        for b in range(B):
            subj_idx = subject_indices[b]
            subj_poses = []
            for s in range(S):
                pose = poses_batch[b][s].copy()
                
                # Convert delta axis-angle to R matrix
                R_delta = axis_angle_to_matrix(self.delta_r[subj_idx, s])
                T_delta = self.delta_t[subj_idx, s]
                
                # Apply to affine: world = A @ img
                # A_refined = [R_delta | T_delta] @ A ? No, simpler is to just adjust points.
                # Let's add the deltas to the pose dict for the rasterizer to use.
                pose['R_delta'] = R_delta
                pose['T_delta'] = T_delta
                subj_poses.append(pose)
            refined_poses.append(subj_poses)
            
        return refined_poses
