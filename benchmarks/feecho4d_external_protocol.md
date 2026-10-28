# FeEcho4D External Radial-View Evaluation

**Status: protocol frozen before scoring (2026-10-23).** This is a separate,
exploratory external-cohort evaluation. It does not alter the MITEA v1/v2
protocols or their reported estimates.

## Scope

FeEcho4D provides annotated radial 2D slices and meshes reconstructed from
those annotations, rather than native dense 3D labels with a complete spatial
transform. The distributed archive also omits the per-subject rotation-center
file used during radial reslicing. Therefore this evaluation measures
completion of withheld radial annotation views in index/angle coordinates; it
does not claim Cartesian 3D reconstruction, physical surface accuracy, or
clinical validation.

## Cohort and target

- Use the 52 annotated subjects only. The 50 unannotated subjects are excluded.
- Evaluate one frame per subject: the `ED_time` in that subject's `_info.cfg`.
- Collapse mask values 127 (LV cavity) and 255 (myocardium) to foreground;
  value 0 is background. No image intensities are used.
- Subject is the statistical unit. Frames, pixels, and radial views are not
  independent observations.
- Do not use FeEcho4D to train, initialize, or select any population model or
  hyperparameter. All 52 subjects remain external evaluation subjects.

## Sparse-view task

The dataset supplies 37 radial views at 5-degree increments from 0 to 180
degrees. Use views 7, 19, and 31 (30, 90, and 150 degrees; one-based file
indices) as the only fitting observations. Score the other 34 views. This
uses three interior angles rather than the geometrically duplicated
0/180-degree endpoint pair as fitting views. The selection is fixed from the
documented angular grid, without inspecting subject masks. Both endpoint
annotations remain among the 34 scored views; view-level scores are therefore
correlated and the subject remains the sole statistical unit.

Represent a pixel by normalized image coordinates `(x,y)` in `[-1,1]` and
the normalized radial angle `a = angle_degrees / 90 - 1`. This coordinate is
an index-space parameterization of the radial-view stack, not a physical XYZ
coordinate system. The fitting code receives only coordinates and binary
labels from the three selected views. Full annotations are read for scoring
after fitting.

## Methods and settings

- **INR (primary):** the repository's Fourier-coordinate occupancy network;
  hidden width 64, four layers, four Fourier frequencies.
- **Gaussian occupancy (secondary):** the repository's 256-primitive field,
  uniform initialization, as selected for MITEA v1.
- Both use Adam, binary cross-entropy, uniform sampling from observed pixels,
  batch size 4,096, 1,000 steps, learning rate 0.001, threshold 0.5, and seeds
  17, 23, and 42. No early stopping, postprocessing, or FeEcho4D-based tuning.
- Fit both fields from scratch for each subject; no MITEA-trained weights or
  population models are transferred.
- Report a run as failed if it produces non-finite values or cannot complete;
  retain failures in the denominator and report their count.

## Outcomes and analysis

- Primary: pooled pixel Dice over all 34 withheld views, computed per subject.
- Secondary: per-subject mean view Dice, IoU, pixel accuracy, and method runtime.
- Average the three seeds within each subject before method comparisons.
- Report subject-level median/IQR and mean/SD. Compare Gaussian with INR using
  paired subject-level differences and a two-sided Wilcoxon signed-rank test;
  provide a 95% percentile bootstrap interval for the paired mean difference
  (10,000 subject resamples, RNG seed 2027). This external analysis is
  descriptive and exploratory.
- Do not compare its scores numerically with MITEA 3D unobserved-voxel Dice.

## Data availability and attribution

Read the archive directly; do not extract it into the repository or commit
derived source images/masks. FeEcho4D's project page specifies noncommercial
research use. Cite the FeEcho4D dataset and its associated publication in any
public report, and retain the source terms with any redistribution decision.


## Execution log and results (2026-10-23)

- `runs/feecho4d_external_20261023`: all 312 runs failed before fitting
  (`_info.cfg` has no INI section header). No masks were scored. The parser was
  fixed; protocol settings were not changed.
- `runs/feecho4d_external_20261023_v2`: an interrupted partial run (8 records),
  superseded and not analysed.
- `runs/feecho4d_external_20261023_final`: the analysed run. It read an
  extracted copy under the git-ignored `dataset/FeEcho4D_extracted` (deviation:
  the protocol said to read the ZIP directly). Nothing derived is committed.
  Completed 312/312 runs (52 subjects × 2 methods × 3 seeds) with 0 failures.
  Summary: `summary_audit.json` (`scripts/summarize_feecho4d.py`).

| Method | Pooled Dice, mean ± SD | Median [IQR] | Fit time (s) |
| --- | --- | --- | --- |
| INR | 0.918 ± 0.015 | 0.920 [0.908, 0.930] | 2.6 |
| Gaussian | 0.805 ± 0.029 | 0.802 [0.779, 0.828] | 6.0 |

Paired Gaussian − INR difference in pooled Dice: mean −0.113 (95% bootstrap CI
−0.120 to −0.105), Wilcoxon p = 3.5 × 10⁻¹⁰. The Gaussian score was lower in
all 52 subjects. Mean-view Dice gives the same result (−0.135, CI −0.143 to
−0.126). The INR scored 0.919 in healthy subjects (n = 47) and 0.912 in CHD
subjects (n = 5). The analysis is exploratory and is not comparable with the
MITEA 3D Dice values.
