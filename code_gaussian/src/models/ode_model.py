import torch
import torch.nn as nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    def __init__(self, latent_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, t, x):
        return self.net(x)

class CardiacODE(nn.Module):
    def __init__(self, latent_dim):
        super(CardiacODE, self).__init__()
        self.ode_func = ODEFunc(latent_dim)
        self.latent_dim = latent_dim

    def forward(self, z0, t):
        """
        z0: initial latent state at t=0 (End-Diastole)
        t: time points to evaluate at (e.g., [0, 1] where 1 is End-Systole)
        """
        zt = odeint(self.ode_func, z0, t, method='rk4')
        return zt

class LatentEncoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=128):
        super(LatentEncoder, self).__init__()
        # Simple convolutional encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)
