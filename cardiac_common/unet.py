"""3D U-Net that completes occupancy from sparse sections (protocol v2).

Input channels in the crop frame: observed label value and observation mask.
Training uses full labels of training subjects only, with the two protocol
layouts and plane positions jittered by up to ``JITTER`` voxels per plane.
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cardiac_common.population import from_crop, observation_channels, to_crop
from cardiac_common.sparse import select_planes

JITTER = 6
LAYOUTS = (("orthogonal", None), ("axial", (0.375, 0.5, 0.625)))


def _block(inputs, outputs):
    return nn.Sequential(
        nn.Conv3d(inputs, outputs, 3, padding=1), nn.InstanceNorm3d(outputs, affine=True), nn.LeakyReLU(0.01),
        nn.Conv3d(outputs, outputs, 3, padding=1), nn.InstanceNorm3d(outputs, affine=True), nn.LeakyReLU(0.01))


class UNet3D(nn.Module):
    def __init__(self, base=16, levels=4):
        super().__init__()
        widths = [base * 2 ** i for i in range(levels)]
        self.down = nn.ModuleList([_block(2 if i == 0 else widths[i - 1], w) for i, w in enumerate(widths)])
        self.up = nn.ModuleList([nn.ConvTranspose3d(widths[i + 1], widths[i], 2, stride=2)
                                 for i in reversed(range(levels - 1))])
        self.merge = nn.ModuleList([_block(2 * widths[i], widths[i]) for i in reversed(range(levels - 1))])
        self.head = nn.Conv3d(widths[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else F.max_pool3d(x, 2))
            skips.append(x)
        for up, merge, skip in zip(self.up, self.merge, reversed(skips[:-1])):
            x = merge(torch.cat([up(x), skip], dim=1))
        return self.head(x)


def observe(mask, planes):
    """Observation mask and in-plane labels for ``planes`` of a native-grid mask."""
    observed = np.zeros(mask.shape, dtype=bool)
    for axis, index in planes:
        selection = [slice(None)] * 3
        selection[axis] = index
        observed[tuple(selection)] = True
    return observed, np.where(observed, mask, False)


def jittered_planes(shape, layout, fractions, rng):
    planes = select_planes(shape, 3, layout, fractions)
    moved = [(axis, int(np.clip(index + rng.integers(-JITTER, JITTER + 1), 0, shape[axis] - 1)))
             for axis, index in planes]
    # Keep distinct parallel planes; fall back to the protocol planes otherwise.
    return moved if len(set(moved)) == len(moved) else planes


def train_unet(masks, iterations, lr, seed, device, log_every=500, snapshots=()):
    """Train on native-grid training masks.

    Returns (model, per-interval mean losses, seconds, {step: state_dict} for
    ``snapshots``). With a constant learning rate, the snapshot at step n equals
    a separate n-step run with the same seed.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(device)
    model = UNet3D().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    targets = [torch.from_numpy(to_crop(mask).astype(np.float32)) for mask in masks]
    history, running, saved, started = [], [], {}, time.perf_counter()
    model.train()
    for step in range(1, iterations + 1):
        index = int(rng.integers(len(masks)))
        layout, fractions = LAYOUTS[int(rng.integers(len(LAYOUTS)))]
        observed, values = observe(masks[index], jittered_planes(masks[index].shape, layout, fractions, rng))
        inputs = torch.from_numpy(observation_channels(observed, values))[None].to(device)
        target = targets[index][None, None].to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(inputs)
        logits = logits.float()
        probability = torch.sigmoid(logits)
        dice = 1 - (2 * (probability * target).sum() + 1) / (probability.sum() + target.sum() + 1)
        loss = F.binary_cross_entropy_with_logits(logits, target) + dice
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running.append(float(loss.detach()))
        if step in snapshots:
            saved[step] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if step % log_every == 0:
            history.append(float(np.mean(running)))
            running = []
            print(f"  step {step}/{iterations}: loss={history[-1]:.4f} "
                  f"({time.perf_counter() - started:.0f}s)", flush=True)
    return model, history, time.perf_counter() - started, saved


@torch.no_grad()
def predict_unet(model, observed, values, shape, device):
    """Native-grid occupancy probability; voxels outside the crop are 0."""
    model.eval()
    inputs = torch.from_numpy(observation_channels(observed, values))[None].to(device)
    with torch.autocast(torch.device(device).type, enabled=torch.device(device).type == "cuda"):
        probability = torch.sigmoid(model(inputs).float())[0, 0].cpu().numpy()
    return from_crop(probability.astype(np.float32), shape)
