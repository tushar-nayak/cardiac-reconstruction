"""INR initialised from the population shape model (protocol v2, `prior_inr`).

1. Fit the shape model to the observed sections (``ssm.fit_ssm``) to get a
   prior occupancy for this subject; no unobserved label is read.
2. Distil: train the v1 INR to reproduce that prior occupancy on uniformly
   sampled grid voxels.
3. Fine-tune on the observed sections with v1's minibatch sequence and step
   budget, optionally adding ``prior_weight`` times the BCE to the prior on
   uniformly sampled unobserved voxels.
"""

import time

import torch
from torch.nn import functional as F

from cardiac_common.sparse import coordinates

DISTILL_STEPS = 1000
DISTILL_LR = 1e-3


def _uniform_batch(shape, count, generator, exclude=None):
    """Random grid indices (zyx); with ``exclude``, rejects observed voxels."""
    sizes = torch.tensor(shape)
    indices = (torch.rand(count, 3, generator=generator) * sizes).long()
    if exclude is not None:
        indices = indices[~exclude[tuple(indices.T)]]
    return indices


def fit_prior_inr(model, points, labels, observed, prior_mask, steps, lr, batch_size, prior_weight,
                  seed, device):
    """Distil then fine-tune ``model`` in place; returns (seconds, final section BCE)."""
    shape = tuple(prior_mask.shape)
    prior = torch.as_tensor(prior_mask, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    started = time.perf_counter()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=DISTILL_LR)
    for _ in range(DISTILL_STEPS):
        indices = _uniform_batch(shape, batch_size, generator)
        prediction = model(coordinates(indices, shape).to(device)).reshape(-1)
        loss = F.binary_cross_entropy(prediction, prior[tuple(indices.T)].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    distill_loss = float(loss.detach())

    # Same section minibatch sequence as v1 ``fit_field`` for this seed.
    sections = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        chosen = torch.randint(len(points), (min(batch_size, len(points)),), generator=sections)
        prediction = model(points[chosen].to(device)).reshape(-1)
        loss = F.binary_cross_entropy(prediction, labels[chosen].to(device))
        section_loss = loss
        if prior_weight > 0:
            indices = _uniform_batch(shape, batch_size, generator, exclude=observed)
            guess = model(coordinates(indices, shape).to(device)).reshape(-1)
            loss = loss + prior_weight * F.binary_cross_entropy(guess, prior[tuple(indices.T)].to(device))
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at fine-tuning step {step + 1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - started, float(section_loss.detach()), distill_loss
