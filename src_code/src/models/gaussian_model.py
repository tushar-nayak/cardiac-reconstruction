import torch
import torch.nn as nn
import torch.nn.functional as F
from src_code.src.occupancy import evaluate_gaussian_occupancy

class GaussianModel(nn.Module):
    def __init__(self, num_gaussians=1000, latent_dim=128):
        super(GaussianModel, self).__init__()
        self.num_gaussians = num_gaussians
        
        # Initial parameters for Gaussians
        self.means = nn.Parameter(torch.randn(num_gaussians, 3) * 50.0)
        self.scales = nn.Parameter(torch.ones(num_gaussians, 3) * 2.0)
        self.rotations = nn.Parameter(torch.tensor([1., 0., 0., 0.]).repeat(num_gaussians, 1))
        
        # VERY low initial raw opacity to start from "empty"
        self.opacities = nn.Parameter(torch.ones(num_gaussians, 1) * -5.0) 
        self.intensities = nn.Parameter(torch.ones(num_gaussians, 1) * 0.5)
        self.semantic_embeddings = nn.Parameter(torch.randn(num_gaussians, 512) * 0.01)

        # MLP to predict deformations
        self.deformation_net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_gaussians * 8) 
        )

    def initialize_from_voxels(self, occupancy_volumes, affines):
        """
        Initialize Gaussian means and opacities from occupied voxels.
        """
        device = self.means.device
        all_pts_world = []
        for i in range(len(occupancy_volumes)):
            occ_indices = torch.nonzero(occupancy_volumes[i])
            if len(occ_indices) > 0:
                num_to_sample = min(len(occ_indices), self.num_gaussians // len(occupancy_volumes))
                selected_indices = occ_indices[torch.randint(0, len(occ_indices), (num_to_sample,))]
                pts_img = selected_indices[:, [2, 1, 0]].float().to(device)
                pts_h = torch.cat([pts_img, torch.ones(len(pts_img), 1).to(device)], dim=-1)
                pts_world = (pts_h @ affines[i].to(device).T)[:, :3]
                all_pts_world.append(pts_world)
        
        if len(all_pts_world) > 0:
            init_pts = torch.cat(all_pts_world, dim=0)
            num_init = len(init_pts)
            
            if num_init < self.num_gaussians:
                extra = torch.randn(self.num_gaussians - num_init, 3).to(device) * 50.0
                init_pts = torch.cat([init_pts, extra], dim=0)
            else:
                init_pts = init_pts[:self.num_gaussians]
                num_init = self.num_gaussians
            
            self.means.data.copy_(init_pts)
            
            # Initialize opacities: high for seeded points, low for extra
            new_opacities = torch.ones(self.num_gaussians, 1, device=device) * -5.0
            new_opacities[:num_init] = 5.0
            self.opacities.data.copy_(new_opacities)
            
            print(f"Initialized {self.num_gaussians} Gaussians from occupancy volumes with opacity seeding.")

    def forward(self, z):
        """
        Returns the deformed Gaussian parameters.
        """
        B = z.shape[0]
        params = self.deformation_net(z).view(B, self.num_gaussians, 8)
        
        delta_means = params[..., :3] * 1.0 
        delta_scales = params[..., 3:6]
        delta_opacities = params[..., 6:7]
        delta_semantics = params[..., 7:8]

        current_means = self.means.unsqueeze(0) + delta_means
        # Keep learned base scales positive even if an optimizer step crosses
        # zero; the previous log(self.scales) formulation could create NaNs.
        base_scales = F.softplus(self.scales).unsqueeze(0) + 1e-4
        current_scales = torch.exp(torch.log(base_scales) + delta_scales * 0.1)
        current_opacities = torch.sigmoid(self.opacities.unsqueeze(0) + delta_opacities)
        current_semantics = self.semantic_embeddings.unsqueeze(0) + delta_semantics
        
        return {
            'means': current_means,
            'scales': current_scales,
            'rotations': self.rotations,
            'opacities': current_opacities,
            'intensities': torch.sigmoid(self.intensities),
            'semantics': current_semantics
        }

    def evaluate_occupancy(self, points, gaussian_params, chunk_size=512):
        """
        Non-saturating occupancy aggregation:
        Soft occupancy = 1 - exp(-sum(alpha_i * weight_i))
        """
        return evaluate_gaussian_occupancy(
            points,
            gaussian_params['means'],
            gaussian_params['scales'],
            gaussian_params['opacities'],
            chunk_size=chunk_size,
        )
