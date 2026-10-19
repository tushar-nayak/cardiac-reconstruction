# Gaussian sparse-plane optimization pilot

The initial Gaussian fit underestimates foreground. On the observed planes of
`MITEA_071`, seed 17, its precision was 0.9981 and recall was 0.9070 at threshold
0.5. It predicted 164.771 mL against 199.527 mL in the complete target.
This suggested testing optimization and initialization before increasing model
size. The comparison below concerns the shared sparse-mask benchmark, not the
older encoder experiment reported on the Gaussian website page.

## Changes

The new optional `--gaussian-init observed` seeds centers from foreground
coordinates on the observed planes, adds normalized-coordinate Gaussian jitter
with standard deviation 0.05, and clips initial positions to `[-1,1]`. If no
observed foreground exists, initialization falls back to uniform positions.
No unobserved labels are used. Scales, opacities, field definition, loss, number
of primitives, and minibatch sampling remain the same.

The tested recipe combines this initialization with Adam learning rate 0.003.
Uniform initialization and learning rate 0.001 remain the command-line defaults
so the original baseline can be reproduced.

## Single-subject ablations

All rows use `MITEA_071`, seed 17, three central orthogonal planes, 256 Gaussians,
batch size 4096, threshold 0.5, and native-resolution evaluation.

| Initialization | Learning rate | Steps | Whole-volume Dice | Unobserved Dice | Volume error (mL) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform | 0.001 | 1000 | 0.9018 | 0.8987 | 34.756 |
| Uniform | 0.003 | 1000 | 0.9240 | 0.9211 | 25.652 |
| Observed foreground | 0.001 | 1000 | 0.9256 | 0.9223 | 24.252 |
| Observed foreground | 0.003 | 1000 | 0.9306 | 0.9272 | 20.559 |
| Observed foreground | 0.003 | 3000 | 0.9144 | 0.9095 | 24.508 |

Longer fitting is not automatically better: the final minibatch BCE fell from
0.0332 to 0.0210 in the 3000-step run while unobserved reconstruction worsened.
Sparse-plane fitting leaves ambiguity away from the supervised planes.

## Matched validation pilot

Control: uniform initialization, learning rate 0.001. Candidate: observed
initialization, learning rate 0.003. Both use 1000 steps and 256 Gaussians
(1792 parameters). Validation indices 0, 1, 2 correspond to `MITEA_071`,
`MITEA_018`, and `MITEA_052`, with seeds 17, 23, and 42 for each subject.
Subject splits use seed 17. Supervision fingerprints, planes, thresholds,
evaluation strides, parameter counts, and update counts were checked to match
for every pair.

| Metric, mean over 9 subject/seed runs | Control | Candidate |
| --- | ---: | ---: |
| Whole-volume Dice | 0.8907 | 0.9174 |
| Unobserved-voxel Dice | 0.8877 | 0.9138 |
| Absolute volume error (mL) | 60.210 | 39.676 |
| Closed prediction surfaces | 9 / 9 | 9 / 9 |

| Subject | Control Dice, mean over seeds | Candidate Dice, mean over seeds |
| --- | ---: | ---: |
| MITEA_071 | 0.8986 | 0.9326 |
| MITEA_018 | 0.8847 | 0.8963 |
| MITEA_052 | 0.8887 | 0.9234 |

Whole-volume Dice improved in all 9 pairs, by 0.0068 to 0.0414. Mean absolute
integrated-curvature difference remained approximately 8.3776 in both groups:
these shape improvements did not change the predicted global topology.
This is a small validation pilot. The settings were first explored on
`MITEA_071`; no test subjects were used and there is no claim of a dataset-wide
win or statistical significance. The pilot does not compare against INR across
these subjects and seeds.

## Reproduce the pilot

```bash
.venv/bin/python scripts/benchmark_sparse.py \
  --data-dir dataset/mitea \
  --run-dir runs/gaussian_uniform_control_pilot_v01 \
  --methods gaussian --plane-layout orthogonal \
  --sample-indices 0 1 2 --seeds 17 23 42 \
  --steps 1000 --num-gaussians 256 --batch-size 4096 \
  --gaussian-init uniform --lr 0.001 \
  --save-occupancy --device cuda

.venv/bin/python scripts/benchmark_sparse.py \
  --data-dir dataset/mitea \
  --run-dir runs/gaussian_observed_lr3e3_pilot_v01 \
  --methods gaussian --plane-layout orthogonal \
  --sample-indices 0 1 2 --seeds 17 23 42 \
  --steps 1000 --num-gaussians 256 --batch-size 4096 \
  --gaussian-init observed --lr 0.003 \
  --save-occupancy --device cuda
```

Each run directory must be new. Raw metrics, configurations, split manifests,
checkpoints, and occupancy tensors are stored in the ignored `runs/` directories.
Per-result metadata records initialization, learning rate, and update count.
Single-subject ablations are saved under `runs/gaussian_lr3e3_v01`,
`runs/gaussian_observed_init_v01`, and `runs/gaussian_observed_3000_v01`;
the original baseline is in `runs/shared_sparse_baseline_v01`.
