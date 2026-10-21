"""Polyharmonic radial-basis implicit surface fitted to sparse sections (protocol v2).

Constraint values are the in-plane signed distances of observed voxels, so the
method is defined for any set of planes, not only orthogonal or parallel ones.
Positions are in voxels (1 mm) scaled by ``SCALE`` for conditioning; the
smoothing weight is in these scaled units.
"""

import numpy as np
import torch

from cardiac_common.interpolation import signed_distance_2d

SCALE = 0.01  # voxel (mm) coordinates -> units of 100 mm
CLIP_MM = 10.0
NEAR_MM = 3.0
MAX_NEAR = 3000
MAX_FAR = 1000
KERNELS = {"r": lambda r: r, "r3": lambda r: r ** 3}


def section_constraints(sections, shape):
    """(zyx positions, clipped in-plane signed distance) for observed voxels.

    Voxels shared by two sections take the mean of their in-plane values.
    """
    totals, counts = {}, {}
    for axis, index, mask in sections:
        field = np.clip(signed_distance_2d(mask), -CLIP_MM, CLIP_MM)
        grid = np.indices(field.shape).reshape(2, -1).T
        positions = np.insert(grid, axis, index, axis=1)
        for key, value in zip(map(tuple, positions), field.reshape(-1)):
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    positions = np.array(list(totals), dtype=np.float64)
    values = np.array([totals[k] / counts[k] for k in totals], dtype=np.float64)
    return positions, values


def choose_centres(values, seed=2027):
    """Indices of near-boundary voxels (at most MAX_NEAR) plus MAX_FAR others."""
    rng = np.random.default_rng(seed)
    near = np.flatnonzero(np.abs(values) <= NEAR_MM)
    far = np.flatnonzero(np.abs(values) > NEAR_MM)
    near = rng.choice(near, min(len(near), MAX_NEAR), replace=False)
    far = rng.choice(far, min(len(far), MAX_FAR), replace=False)
    return np.sort(np.concatenate([near, far]))


def fit_rbf(positions, values, kernel="r", smoothing=0.0):
    """Solve [Φ + λI, P; Pᵀ, 0][w; c] = [f; 0] with a linear polynomial P."""
    x = positions * SCALE
    phi = KERNELS[kernel](np.linalg.norm(x[:, None] - x[None], axis=-1))
    phi[np.diag_indices_from(phi)] += smoothing
    poly = np.hstack([np.ones((len(x), 1)), x])
    system = np.block([[phi, poly], [poly.T, np.zeros((4, 4))]])
    solution = np.linalg.solve(system, np.concatenate([values, np.zeros(4)]))
    return {"centres": x, "weights": solution[:len(x)], "poly": solution[len(x):], "kernel": kernel}


def evaluate_rbf(fit, shape, device="cpu", chunk=65536):
    """Interpolant on the full zyx grid, float64 throughout."""
    device = torch.device(device)
    centres = torch.as_tensor(fit["centres"], device=device)
    weights = torch.as_tensor(fit["weights"], device=device)
    poly = torch.as_tensor(fit["poly"], device=device)
    kernel = KERNELS[fit["kernel"]]
    grid = np.indices(shape).reshape(3, -1).T
    output = np.empty(len(grid))
    for start in range(0, len(grid), chunk):
        x = torch.as_tensor(grid[start:start + chunk] * SCALE, device=device)
        value = kernel(torch.cdist(x, centres)) @ weights + poly[0] + x @ poly[1:]
        output[start:start + chunk] = value.cpu().numpy()
    return output.reshape(shape)


def rbf_complete(sections, shape, kernel="r", smoothing=0.0, device="cpu"):
    """Foreground mask where the fitted interpolant is >= 0, plus centre count."""
    positions, values = section_constraints(sections, shape)
    chosen = choose_centres(values)
    fit = fit_rbf(positions[chosen], values[chosen], kernel, smoothing)
    return evaluate_rbf(fit, shape, device) >= 0, len(chosen)
