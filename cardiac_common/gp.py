"""Gaussian-process implicit surface fitted to sparse sections (protocol v2).

Same constraints and centres as ``rbf`` (in-plane signed distances of observed
voxels, clipped to ±10 mm). GP regression with a Matérn-5/2 kernel and a
constant prior mean: far from the sections the posterior mean reverts to the
prior mean, so a negative prior mean says "outside unless observed otherwise".
The posterior standard deviation is the uncertainty map.
"""

import numpy as np
import torch

from cardiac_common.rbf import choose_centres, section_constraints

NOISE_MM = 0.5  # section signed distances are quantised at the 1 mm voxel scale
VARIANCE_STRIDE = 2


def matern52(distance, length):
    scaled = np.sqrt(5.0) * distance / length
    return (1 + scaled + scaled ** 2 / 3) * torch.exp(-scaled)


def fit_gp(positions, values, length, prior_mean, device="cpu"):
    """Posterior weights for zero-mean residuals ``values - prior_mean``.

    The signal variance is the mean squared residual, so the posterior mean
    does not depend on it and the posterior SD is in mm.
    """
    device = torch.device(device)
    x = torch.as_tensor(positions, device=device, dtype=torch.float64)
    residual = torch.as_tensor(values - prior_mean, device=device, dtype=torch.float64)
    variance = float((residual ** 2).mean().clamp_min(1.0))
    gram = variance * matern52(torch.cdist(x, x), length)
    gram.diagonal().add_(NOISE_MM ** 2)
    cholesky = torch.linalg.cholesky(gram)
    alpha = torch.cholesky_solve(residual[:, None], cholesky)[:, 0]
    return {"centres": x, "alpha": alpha, "cholesky": cholesky, "variance": variance,
            "length": length, "prior_mean": prior_mean}


def _grid(shape, stride, device):
    axes = [torch.arange(0, n, stride, device=device, dtype=torch.float64) for n in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


@torch.no_grad()
def posterior_mean(fit, shape, chunk=65536, exact=False):
    """Posterior mean on the full grid. Kernel values are computed in float32
    on coordinates centred at the array centre (float64 is many times slower
    on consumer GPUs) and summed in float64."""
    dtype = torch.float64 if exact else torch.float32
    centre = torch.tensor([(n - 1) / 2 for n in shape], dtype=torch.float64, device=fit["centres"].device)
    centres = (fit["centres"] - centre).to(dtype)
    grid = (_grid(shape, 1, centres.device) - centre).to(dtype)
    output = torch.empty(len(grid), dtype=torch.float64, device=grid.device)
    for start in range(0, len(grid), chunk):
        k = matern52(torch.cdist(grid[start:start + chunk], centres), fit["length"]).double()
        output[start:start + chunk] = fit["prior_mean"] + fit["variance"] * (k @ fit["alpha"])
    return output.reshape(shape).cpu().numpy()


@torch.no_grad()
def posterior_sd(fit, shape, stride=VARIANCE_STRIDE, chunk=16384):
    """Posterior SD (mm) on the grid ``[::stride, ::stride, ::stride]``.

    The inverse Cholesky factor is formed in float64 and applied in float32,
    which is ample for an uncertainty map and much faster on the GPU.
    """
    centres = fit["centres"]
    identity = torch.eye(len(centres), dtype=torch.float64, device=centres.device)
    inverse = torch.linalg.solve_triangular(fit["cholesky"], identity, upper=False).float()
    centre = torch.tensor([(n - 1) / 2 for n in shape], dtype=torch.float64, device=centres.device)
    grid = (_grid(shape, stride, centres.device) - centre).float()
    centres = (centres - centre).float()
    output = torch.empty(len(grid), dtype=torch.float32, device=grid.device)
    for start in range(0, len(grid), chunk):
        k = fit["variance"] * matern52(torch.cdist(grid[start:start + chunk], centres), fit["length"])
        whitened = inverse @ k.T
        output[start:start + chunk] = (fit["variance"] - (whitened ** 2).sum(0)).clamp_min(0).sqrt()
    return output.reshape([len(range(0, n, stride)) for n in shape]).cpu().numpy()


def gp_complete(sections, shape, length, prior_mean, device="cpu", uncertainty=True):
    """(foreground mask, strided posterior mean, strided posterior SD or None, centre count)."""
    positions, values = section_constraints(sections, shape)
    chosen = choose_centres(values)
    fit = fit_gp(positions[chosen], values[chosen], length, prior_mean, device)
    mean = posterior_mean(fit, shape)
    sd = posterior_sd(fit, shape) if uncertainty else None
    return mean >= 0, mean[::VARIANCE_STRIDE, ::VARIANCE_STRIDE, ::VARIANCE_STRIDE], sd, len(chosen)
