# Restoring and building on the May Gaussian recipe

The reference comes from commit `761ef2df119a1fa47a79e701e5e1ea8179cbcfd6`
(May 15, 2026), where direct subject fitting and its mesh viewer were added.
The May 16 documentation snapshot is `7bb2a3860ff7698ce24d101aab7b4420007035f4`.
The historical fitted checkpoint is not present locally or tracked in git;
this is a controlled recipe reproduction with seed 17, not recovery of the
original weights or exact random sample.

## What produced the earlier result

- 1,800 axis-aligned Gaussian primitives in physical coordinates.
- Centers sampled from occupied voxels throughout the full 3D label, plus
  0.4 mm Gaussian jitter.
- Initial per-axis scales `softplus(0.9) + 0.35`, approximately 1.591 mm;
  initial opacity `sigmoid(1.4)`, approximately 0.802.
- Full-volume point supervision: half jittered foreground samples, half uniform
  samples, with BCE weights 1.5 and 2.0 respectively.
- Adam learning rate 0.003, 4096 points per step, scale penalty 0.002, opacity
  penalty 0.01, and 600 documented fit steps.
- Historical sorted subject splits: first 110 train, next 15 validation,
  remainder test. On this dataset, validation index 0 is `MITEA_111`.

This differs substantially from the recent 256-Gaussian, three-plane benchmark.
For example, the latter starts with much broader effective physical kernels
(about 9–11 mm for `MITEA_071`). The old result also reported an IoU over sampled
points concentrated in foreground/centroid regions, rather than all voxels.

## Reproduction and refinement

All rows below use native-grid evaluation, threshold 0.5, full-label supervision,
1,800 primitives, and seed 17. The refinement changes only the fit step budget
from 600 to 1800. It receives three times as many updates.

| Subject | Steps | Dice | IoU | Absolute volume error (mL) | Mesh components: prediction / target |
| --- | ---: | ---: | ---: | ---: | ---: |
| MITEA_111 | 600 | 0.9449 | 0.8956 | 20.900 | 37 / 4 |
| MITEA_111 | 1800 | 0.9776 | 0.9562 | 11.219 | 1 / 4 |
| MITEA_071 | 600 | 0.9678 | 0.9377 | 3.346 | 7 / 2 |
| MITEA_071 | 1800 | 0.9812 | 0.9632 | 7.116 | 1 / 2 |

The reproduced historical sampled IoU on `MITEA_111` is 0.9487, versus the
archived 0.9503. At 1800 steps it is 0.9981, while native-grid IoU is 0.9562:
the sampled diagnostic remains more optimistic. It must not replace whole-grid
evaluation or be presented as sparse-view reconstruction accuracy.

Refinement improves overlap and reduces fragments in both examples. It does
not recover every target component, and the volume error worsens for
`MITEA_071` despite better overlap. For `MITEA_111`, ΔK changes from about 339.292
at 600 steps to -12.566 at 1800 steps. For `MITEA_071`, it changes from 75.398
to approximately zero, although the prediction and target still have different
component counts. All extracted components, regardless of size, are counted.

These are fits to known labels on two subjects, with no population prior or
held-out-view reconstruction claim. Strong full-label fitting establishes a
representation reference; using it to learn a prior from training subjects for
later sparse-view adaptation is a subsequent experiment.

## Reproduce

```bash
.venv/bin/python code_gaussian/scripts/fit_subject_reconstruction.py \
  --data-dir dataset/mitea \
  --run-dir runs/gaussian_may2026_reference_v01 \
  --recipe may2026 --evaluate --device cuda

.venv/bin/python code_gaussian/scripts/fit_subject_reconstruction.py \
  --data-dir dataset/mitea \
  --run-dir runs/gaussian_may2026_refined_v01 \
  --recipe may2026 --steps 1800 --evaluate --device cuda
```

For `MITEA_071`, add `--split-policy shuffled` and choose separate new run
directories. The results recorded here use `runs/gaussian_may2026_MITEA071_v01`
and `runs/gaussian_may2026_MITEA071_refined_v01`. Every checkpoint records the
subject, volume shape, affine, full-label supervision, and split policy.
The evaluation/gallery scripts honor that policy and reject a subject mismatch.
Other workflows retain the default shuffled dataset split.

Export the two-subject website assets after all four runs complete:

```bash
.venv/bin/python scripts/export_gaussian_reference.py
python3 -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Open `http://localhost:8000/gaussian.html`. Generated assets must go
into a new directory. The results page offers native-resolution slice errors,
synchronized meshes, reference overlays, and per-subject geometry diagnostics.
The published Gaussian page presents the Previous Stable Recipe and refined
fits. The experimental sparse-plane pilot and earlier encoder assets are kept
outside the deployed `docs/` tree.
