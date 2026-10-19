# Initial shared sparse-plane baseline

This is a preliminary check on one validation subject, not a dataset-wide
comparison or an echo-image reconstruction result.

Both fields fit the binary foreground masks on the same three central
orthogonal planes. The loader defines foreground as all labels greater than
zero. Plane locations depend only on array dimensions; neither method uses
unobserved labels for training or initialization. INR uses the existing
positional-encoding MLP; the Gaussian field uses 256 axis-aligned primitives
initialized uniformly in normalized coordinates.

## Reproduce

```bash
.venv/bin/python scripts/benchmark_sparse.py \
  --data-dir dataset/mitea \
  --run-dir runs/shared_sparse_baseline_v01 \
  --plane-layout orthogonal \
  --steps 1000 \
  --batch-size 4096 \
  --num-gaussians 256 \
  --save-occupancy \
  --device cuda
```

Use a new run directory when repeating an experiment. This run used seed 17,
split seed 17, validation index 0 (`MITEA_071`, scan 1 ED), Adam with learning
rate 0.001, and native-resolution evaluation at threshold 0.5.
The volume shape in ZYX order was `(146, 140, 122)`. Observed planes were
`z=72`, `y=69`, and `x=60`: 54,925 distinct observed voxels. Both runs recorded
the supervision SHA256
`7bf158ddbcdc09735dc06ef9a932490b9b239d4073aac2b623e60b467d3a37d5`.

| Method | Whole-volume Dice | Unobserved-voxel Dice | Whole-volume IoU | Volume error (mL) | Fit time (s) | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| INR | 0.9360 | 0.9323 | 0.8798 | 9.331 | 4.72 | 9,985 |
| Gaussian occupancy | 0.9018 | 0.8987 | 0.8211 | 34.756 | 7.60 | 1,792 |

Times are wall-clock fit durations on a Quadro RTX 4000 with PyTorch
2.14.1+cu130, including
minibatch transfers and progress reporting. The methods share the update
budget, but differ in parameter count and computation. Timing is sensitive to
hardware, warm-up, and other GPU workloads; it is not a speed benchmark.
Detailed metrics, checkpoints, occupancy grids, and the split manifest remain
in the ignored run directory.

## Next experiments

Evaluate multiple validation subjects and seeds before comparing methods.
Then vary plane count and orientation, add surface-distance metrics, and
measure whether balanced sampling or model-specific improvements help with
unobserved regions. Keep this protocol separate from the original published
INR experiments and full-label Gaussian fits because their supervision and
evaluation settings differ.
