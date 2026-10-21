"""PCA shape model of signed distance maps, fitted to sparse sections (protocol v2).

Training maps share the array-centred crop frame (``population.CROP_SHAPE``)
and are not further aligned; at fit time a 3D translation absorbs residual
offset. Modes are scaled by their standard deviation, so coefficients ``b``
are in units of SD. The fit minimises mean observed-voxel BCE plus
``prior_weight * sum(b**2)``.
"""

import numpy as np
import torch
import torch.nn.functional as F

from cardiac_common.population import (CROP_SHAPE, SDF_CLIP_MM, crop_origin, crop_slices,
                                        signed_distance_3d, to_crop)

TEMPERATURE_MM = 2.0
FIT_STEPS = 300
FIT_LR = 0.05


def build_ssm(masks, subjects, max_modes=40):
    """PCA of crop-frame signed distance maps; returns a serialisable artifact."""
    maps = np.stack([signed_distance_3d(to_crop(mask)).reshape(-1) for mask in masks])
    mean = maps.mean(axis=0)
    centred = torch.from_numpy(maps - mean).double()
    # Eigendecomposition of the small subject-by-subject Gram matrix.
    gram = centred @ centred.T
    eigenvalues, vectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)[:min(max_modes, len(masks) - 1)]
    eigenvalues, vectors = eigenvalues[order], vectors[:, order]
    modes = (centred.T @ vectors) / eigenvalues.sqrt()  # unit-norm columns
    sd = (eigenvalues / (len(masks) - 1)).sqrt()
    return {"mean": torch.from_numpy(mean).half().reshape(CROP_SHAPE),
            "modes": (modes * sd).T.float().half().reshape(-1, *CROP_SHAPE),  # SD-scaled
            "sd": sd.float(), "training_subjects": list(subjects), "crop_shape": CROP_SHAPE}


def _sample(volume, points_zyx, offset):
    """Trilinear samples of ``volume`` (C,D,H,W) at crop positions minus ``offset``."""
    sizes = torch.tensor(volume.shape[1:], device=points_zyx.device, dtype=points_zyx.dtype)
    normalised = 2 * (points_zyx - offset) / (sizes - 1) - 1
    grid = normalised.flip(-1).reshape(1, -1, 1, 1, 3)  # grid_sample wants (x,y,z)
    out = F.grid_sample(volume[None], grid, align_corners=True, padding_mode="border")
    return out.reshape(volume.shape[0], -1)


def fit_ssm(artifact, observed_zyx, labels, shape, modes, prior_weight, device="cpu"):
    """Fit ``modes`` coefficients and a translation; return a native-grid mask."""
    device = torch.device(device)
    mean = artifact["mean"].float().to(device)[None]
    basis = artifact["modes"][:modes].float().to(device)
    start = torch.tensor(crop_origin(shape, artifact["crop_shape"]), device=device, dtype=torch.float32)
    points = torch.as_tensor(observed_zyx, device=device, dtype=torch.float32) - start
    labels = torch.as_tensor(labels, device=device, dtype=torch.float32)
    sizes = torch.tensor(artifact["crop_shape"], device=device, dtype=torch.float32)
    inside = ((points >= 0) & (points <= sizes - 1)).all(dim=1)
    coefficients = torch.zeros(modes, device=device, requires_grad=True)
    offset = torch.zeros(3, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([coefficients, offset], lr=FIT_LR)
    for _ in range(FIT_STEPS):
        optimizer.zero_grad()
        field = _sample(mean, points, offset)[0] + coefficients @ _sample(basis, points, offset)
        field = torch.where(inside, field, torch.full_like(field, -SDF_CLIP_MM))
        loss = (F.binary_cross_entropy_with_logits(field / TEMPERATURE_MM, labels)
                + prior_weight * (coefficients ** 2).sum())
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        grid = torch.stack(torch.meshgrid(*[torch.arange(n, device=device, dtype=torch.float32)
                                            for n in artifact["crop_shape"]], indexing="ij"), -1).reshape(-1, 3)
        field = _sample(mean, grid, offset)[0] + coefficients @ _sample(basis, grid, offset)
        crop = (field >= 0).reshape(artifact["crop_shape"]).cpu().numpy()
    mask = np.zeros(shape, dtype=bool)
    src, dst = crop_slices(shape, artifact["crop_shape"])
    mask[src] = crop[dst]
    return mask, {"coefficients": coefficients.detach().cpu().tolist(),
                  "translation_zyx": offset.detach().cpu().tolist(), "final_loss": float(loss.detach())}
