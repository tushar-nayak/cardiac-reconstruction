# MITEA Sparse-Plane Reconstruction Benchmark Protocol v1

**Status:** frozen before manifest-aware benchmark implementation and cohort
runs. Any protocol deviation must be logged with a dated rationale; the
confirmatory evaluation subjects must not be used to choose or revise settings.

## Objective and scope

Evaluate completion of a binary cardiac foreground volume from sparse,
synthetically selected annotation planes with known positions. The foreground
is the union of MITEA labels 1 (myocardium) and 2 (LV cavity). This is a
label-conditioned geometric completion benchmark: the methods do not receive
ultrasound intensities, estimate plane pose, or reconstruct from raw
echocardiographic images. Claims and the manuscript title/abstract must not
overstate it as image-based or clinical reconstruction.

## Cohort, split, and leakage control

- Use only the frozen subject manifest
  `benchmarks/splits/mitea_cohort_v1.json` (SHA-256 of split payload:
  `363fae95e79bb50694bfbe5000a5f562a2f6629b1bfe4bf8a7dac78d6fb90932`).
- The manifest contains 134 subjects: a held-out test set of 26 and five
  disjoint development folds of sizes 22, 22, 22, 21, and 21. All scans and
  phases of a subject remain assigned together.
- Fold 0 (22 subjects) is for development only: pipeline checks, the
  predeclared Gaussian recipe choice, and any permitted hyperparameter choice
  happen there. The confirmatory development report uses folds 1--4 (86
  subjects). Do not include fold 0 in confirmatory statistics after using it
  for selection. This resolves the plan's conflicting shorthand of reporting
  `n=108`: `n=108` would include the tuning fold and is not the confirmatory
  estimate under this protocol.
- Do **not** run or inspect the 26 test subjects for this arXiv version. They
  remain reserved for a later, separately approved study. No split is
  regenerated from data ordering or the legacy loader's 80/10/10 partition.
- The unit of inference is the subject, not a seed, voxel, slice, or scan.

## Target and input protocol

- Target: `scan1`, end-diastole (ED), labels collapsed as
  `foreground = (label == 1) OR (label == 2)`; background is 0.
- Keep each volume on its native voxel grid. Do not resample, crop, pad, or
  normalize labels for the primary benchmark. Each method is evaluated on the
  same native grid and affine.
- Primary layout: three central orthogonal cross-sections, one per array axis,
  at index `floor((dimension - 1) / 2)`. Secondary layout: three axial
  cross-sections at indices `floor((i + 0.5) * depth / 3)` for `i=0,1,2`.
- The observed mask is the union of selected planes (intersections counted
  once). Plane indices depend only on array shape, never on foreground content
  or held-out labels. The runner may materialize the target to extract plane
  samples, but training, initialization, and selection code may receive only
  coordinates and labels from observed voxels. Evaluation code may access the
  full target only after fitting is complete.
- An experiment fails validation if a method's saved observation fingerprint,
  selected indices, target, or subject differs from its paired methods.

## Methods and fixed settings

Compare three methods under each layout:

1. **INR (primary method):** the repository's Fourier-coordinate occupancy
   network with hidden width 64, four layers, and four Fourier frequencies,
   initialized with the framework default parameters.
2. **Gaussian occupancy (secondary learned method):** the repository's
   256-primitive sparse Gaussian field. The fold-0 development step selects
   between only the two already-tested recipes: uniform centers with Adam
   learning rate `0.001`, or observed-foreground initialization with learning
   rate `0.003` (sample observed foreground centers with coordinate jitter
   standard deviation `0.05`, clipped to `[-1,1]`). The selected recipe is
   fixed before folds 1--4 are run. No full-volume labels may enter
   initialization.
3. **Nonlearned baselines:** for orthogonal planes, the section-extrusion
   intersection (SEI) is the intersection of the three foreground extrusions;
   for axial planes, linearly interpolate signed distance fields between adjacent
   observed slices and extend the nearest observed field to the volume ends.
   Define signed distance as
   `distance_transform_edt(foreground) - distance_transform_edt(background)`,
   positive inside foreground; classify interpolated values `>= 0` as
   foreground. Learned probabilities use threshold `0.5`; SEI outputs are
   binary by construction. SEI is not a standard silhouette visual hull and
   has no general superset/subset or volume-bias guarantee for sectional data.

For INR and Gaussian fitting, use Adam, binary cross-entropy on uniformly
sampled observed-plane voxels, batch size 4,096, 1,000 optimizer steps, and
three seeds (17, 23, 42). Initial learning rate is `0.001`; the Gaussian recipe
selection above is the sole predeclared exception. Do not apply early stopping,
test-label-informed initialization, component filtering, or other postprocessing
to primary predictions. Run largest-connected-component filtering only as a
clearly labeled secondary sensitivity analysis.

The fold-0 Gaussian recipe choice is made from unobserved-voxel Dice: average
seeds within each subject and layout, average the two layout scores per
subject, then average over the 22 subjects. Ties within 0.002 Dice are resolved
by lower mean whole-volume Dice error aggregated in the same way (absolute
difference from 1); if still tied, select the simpler uniform-initialization
recipe. Record the candidate results and selected configuration before any
fold-1--4 result is generated.
No other settings are tuned. If a correctness bug requires a change, stop,
document it, and restart the affected fold-0 selection before proceeding.

## Outcomes and metric definitions

- **Primary:** unobserved-voxel Dice, computed only on the complement of the
  union of observed planes. If both prediction and target are empty in this
  region, score 1; if only one is empty, score 0.
- **Secondary overlap:** whole-volume Dice and IoU at probability/mask threshold
  `0.5`.
- **Secondary geometry:** HD95 (maximum of the two directed 95th percentiles),
  ASSD (mean of both directed surface-distance sets), absolute volume error,
  connected-component count, boundary/nonmanifold mesh diagnostics, and signed
  integrated-Gaussian-curvature difference
  `DeltaK = K_prediction - K_target` when both extracted surfaces are closed
  manifolds. Also report `abs(DeltaK)` descriptively. `DeltaK` is not a
  substitute for local shape distance; report unavailable values as missing,
  never zero. Do not silently cap surfaces touching the grid boundary.
- **Physical calibration:** the NIfTI audit found 1 mm isotropic spacing and
  matching image/label affines for all 536 volumes; the MITEA publication also
  states that 3DE volumes were reconstructed with 1 mm isotropic voxels
  ([Zhao et al., 2022](https://doi.org/10.3389/fcvm.2022.1016703)). Report
  HD95/ASSD in mm and volume error in mL using each NIfTI affine. Preserve this
  provenance in the manuscript and do not infer finer acquisition resolution
  than the source supports.
- Record empty predictions, failed meshes, nonfinite values, runtime, and
  failed runs. Report the denominator for every metric and do not silently
  exclude failures.

## Statistical analysis

- For each method/layout/subject, average the three seed-level scores first.
- Report the 86 subject-level values using median and interquartile range, plus
  mean and standard deviation as descriptive summaries. Show paired subject
  differences against INR for Gaussian occupancy and interpolation.
- Use two-sided paired Wilcoxon signed-rank tests on subject-level differences.
  Apply Holm correction across the four planned comparisons (two methods × two
  layouts); report corrected p-values.
- Compute 95% percentile bootstrap confidence intervals for paired mean
  differences by resampling the 86 subjects with replacement (10,000
  replicates; bootstrap RNG seed 2027). Seeds are averaged before resampling.
- Pathology subgroup results are descriptive only, with subgroup counts and no
  confirmatory significance claims.
- The 26-subject test set is not part of any estimate in this manuscript.

## Artifacts and reporting

Save per-run configuration, subject ID, split/fold, planes, observation
fingerprint, seed, method initialization, optimizer settings, software version,
fit/evaluation time, and all metrics. Save dense predictions only for the
predeclared qualitative subjects MITEA_071, MITEA_018, and MITEA_052, provided
they are in the development evaluation folds; otherwise choose the nearest
eligible subjects by manifest order before looking at their predictions.
Generate aggregate tables and plots from machine-readable per-run records.
Report both layouts separately, and label the Gaussian method as secondary.
Do not describe the benchmark as a clinical validation or a raw-image
reconstruction study.

## Deviation log

### 2026-10-19 — Amendment 1: axial plane positions (before any learned-method fold-0 result)

**Change.** The secondary axial layout places planes at
`floor(f * depth)` for `f = 0.375, 0.5, 0.625` (`--axial-fractions 0.375 0.5 0.625`),
replacing `floor((i + 0.5) * depth / 3)` (fractions 1/6, 1/2, 5/6). Plane
indices still depend only on array shape. The orthogonal primary layout is
unchanged.

**Rationale.** During Phase 1 smoke testing, the frozen axial layout left two of
three planes without foreground in 102 of 108 development subjects (one in the
other 6). The cardiac foreground occupies roughly the central third to
three-quarters of array axis 0, so the frozen layout reduced the secondary
analysis to a single informative section.

**Derivation.** The new fractions were derived from fold 0 only: for each fold-0
subject, the fraction of axis 0 spanned by foreground was measured; planes were
placed at the quartiles of the median extent (0.319–0.750), giving
0.427/0.535/0.642, rounded to the nearest 1/8. With these fractions, 99 of 108
development subjects have no empty plane and 9 have one.

**Disclosure.** Before this derivation, foreground-extent statistics (empty-plane
counts and median axis-0 extent, not method outputs) had been computed over all
108 development subjects, including confirmatory folds 1–4. No method
predictions or scores from folds 1–4 had been generated or inspected. No test
subject was loaded. The only fold-0 outputs at the time were one-subject
(MITEA_007) pipeline smoke checks.

### 2026-10-19 — Clarification 1: signed distance of an empty or full section

The protocol formula `edt(foreground) - edt(background)` is undefined for a
section with no foreground or no background (SciPy returns boundary-dependent
values). An all-background section is assigned the constant signed distance
`-(height + width)` and an all-foreground section `+(height + width)`, in
voxels. Interpolated slices therefore remain foreground only very close to a
nonempty observed section adjacent to an empty one.

### 2026-10-20 — Fold-0 selection: Gaussian recipe frozen

`scripts/summarize_benchmark.py select-gaussian` selected `uniform_lr0.001`
(uniform center initialization, learning rate 0.001) by the primary criterion.
Layout-averaged, seed-averaged unobserved Dice over 22 fold-0 subjects: uniform
0.8545, observed/lr 0.003 0.8043 (difference −0.0501, outside the 0.002 tie
threshold). Per layout: uniform 0.9021 orthogonal / 0.8069 axial; observed
0.9420 orthogonal / 0.6667 axial. Fold 0 ran 22 subjects × 3 seeds per learned
method and layout, with 0 failed runs and 0 empty predictions.

No method, setting, or metric was changed after the fold-0 runs. On the axial
layout INR and interpolation over-extend along axis 0 (median predicted volume
758 and 675 mL against a 297 mL target); this is recorded as an observed property
of the frozen methods, not corrected. Outputs: `runs/protocol_v1_fold0_*`,
`runs/protocol_v1_fold0_gaussian_selection.json`. This entry was written before
any fold-1–4 command was run.

### 2026-10-20 — Confirmatory run completed

Folds 1–4 (86 subjects) were run once with the settings frozen above, using
`runs/protocol_v1_confirmatory_{orthogonal,axial}_{inr_interp,gaussian}`
(start 15:26, end 17:57 CDT). 1,204 runs (258 per learned method and layout, 86
per interpolation layout); 0 failed runs, 0 empty predictions, no interruptions
or reruns. Partial results were not inspected while runs were in progress. The
confirmatory report is `runs/protocol_v1_confirmatory_report.json` (with `.tex`).
The test split was not loaded.

Analyses added after the confirmatory run, and therefore not prespecified:
the MITEA_071 isosurface panel and per-method fitting times (both descriptive).
