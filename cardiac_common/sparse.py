"""Label-independent plane selection and common occupancy evaluation."""

import numpy as np
import torch

from code_gaussian.src.metrics import binary_metrics


def select_planes(shape, count=3, layout="axial"):
    """Return (zyx axis, voxel index) planes using volume dimensions only."""
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError("shape must contain three positive dimensions")
    if count < 1:
        raise ValueError("count must be positive")
    if layout == "orthogonal":
        if count > 3:
            raise ValueError("orthogonal layout supports at most three planes")
        return [(axis, (shape[axis] - 1) // 2) for axis in range(count)]
    if layout != "axial" or count > shape[0]:
        raise ValueError("axial layout requires count <= volume depth")
    # Bin centers; no search for foreground or anatomy in held-out labels.
    indices = np.floor((np.arange(count) + 0.5) * shape[0] / count).astype(int)
    return [(0, int(index)) for index in indices]


def coordinates(indices, shape):
    """Canonical (z,y,x) indices -> normalized (x,y,z) coordinates."""
    xyz = indices[..., [2, 1, 0]].float()
    sizes = xyz.new_tensor(list(reversed(shape)))
    return 2 * xyz / (sizes - 1).clamp_min(1) - 1


def plane_samples(volume, planes):
    """Read supervision only at selected planes, deduplicating intersections."""
    shape = tuple(volume.shape)
    observed = torch.zeros(shape, dtype=torch.bool)
    for axis, index in planes:
        selection = [slice(None)] * 3
        selection[axis] = index
        observed[tuple(selection)] = True
    indices = observed.nonzero()
    return coordinates(indices, shape), volume[observed].float(), observed


def evaluate_grid(model, shape, device, chunk_size=4096, stride=1):
    """Query native voxel positions in bounded chunks; stride>1 is a smoke check."""
    if chunk_size < 1 or stride < 1:
        raise ValueError("chunk_size and stride must be positive")
    axes = [torch.arange(0, size, stride) for size in shape]
    sampled_shape = tuple(len(axis) for axis in axes)
    output = torch.empty(int(np.prod(sampled_shape)))
    model.eval()
    with torch.no_grad():
        for start in range(0, output.numel(), chunk_size):
            linear = torch.arange(start, min(start + chunk_size, output.numel()))
            d, h, w = sampled_shape
            indices = torch.stack((axes[0][linear // (h * w)],
                                   axes[1][(linear // w) % h],
                                   axes[2][linear % w]), dim=-1)
            output[start:start + len(linear)] = model(
                coordinates(indices, shape).to(device)
            ).reshape(-1).cpu()
    return output.reshape(sampled_shape)


def reconstruction_metrics(probability, target, observed, affine, threshold=0.5):
    """Report whole-grid and unobserved-voxel overlap, plus physical volume error."""
    pred = np.asarray(probability) >= threshold
    truth = np.asarray(target) >= 0.5
    seen = np.asarray(observed, dtype=bool)
    full = binary_metrics(pred, truth)
    # Two empty masks agree exactly; avoid labeling an empty background as Dice=0.
    if not pred.any() and not truth.any():
        full.update(dice=1.0, iou=1.0)
    unseen = None
    if (~seen).any():
        unseen = binary_metrics(pred[~seen], truth[~seen])
        if not pred[~seen].any() and not truth[~seen].any():
            unseen.update(dice=1.0, iou=1.0)
    voxel_mm3 = float(abs(np.linalg.det(np.asarray(affine)[:3, :3])))
    predicted_ml = float(pred.sum() * voxel_mm3 / 1000)
    target_ml = float(truth.sum() * voxel_mm3 / 1000)
    return {"whole_grid": full, "unobserved_voxels": unseen,
            "predicted_volume_ml": predicted_ml, "target_volume_ml": target_ml,
            "absolute_volume_error_ml": abs(predicted_ml - target_ml)}
