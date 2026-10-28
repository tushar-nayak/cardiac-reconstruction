# MITEA Sparse-Plane Reconstruction Benchmark Protocol v2

**Status: frozen for one-time held-out test evaluation (2026-10-22).** Fold-0
grids and selections are development evidence. The 26-subject test set has not
been loaded or scored. This protocol supersedes the proposed folds-1–4 v2
confirmatory analysis: those folds were used to train population models during
fold-0 tuning and are not an independent v2 test set. The test set is evaluated
once with the settings below; no tuning or method changes are allowed after
test outcomes are inspected.

## Objective and relation to v1

Protocol v1 compared per-subject methods that use no information beyond the
observed sections (INR, Gaussian occupancy, layout-specific interpolation).
Its confirmatory result was that per-subject fields gain little over
interpolation with well-placed sections, and that a method's implicit prior
on unobserved extent dominates when sections do not bracket the anatomy.
Protocol v2 adds five methods from three families so the benchmark can answer whether:

1. a non-learned interpolant that is not tied to plane orientation matches the
   v1 layout-specific baselines, and
2. population priors learned from other subjects' full labels improve
   completion over per-subject fitting, and
3. a per-subject field initialised from a population prior (`prior_inr`)
   keeps the prior's benefit while fitting the sections.

Everything not changed here is inherited from v1 unchanged: cohort manifest,
target (scan 1, ED, label union), plane layouts (orthogonal central; parallel
axis-0 sections at 0.375/0.5/0.625 per v1 Amendment 1), threshold 0.5,
metrics, missing-value handling, and the subject as the unit of inference.
At the time this protocol was frozen, the 26-subject test split had not been
used; its one-time final evaluation is specified below.

## Methods

All five methods receive only the observed-section coordinates and labels
of the evaluated subject. Population methods additionally receive full labels
of *other* subjects as training data, under the cross-fitting rule below.

1. **`rbf` (non-learned):** a polyharmonic radial-basis implicit surface
   (variational implicit surface). Constraint values are the in-plane signed
   distances of observed voxels, computed per section as in v1
   (`edt(foreground) - edt(background)`, positive inside, empty/full sections
   per v1 Clarification 1), clipped to ±10 mm. Centers: all observed voxels
   with |signed distance| ≤ 3 mm, subsampled uniformly to at most 3,000, plus
   1,000 further observed voxels sampled uniformly; sampling RNG seed 2027.
   Interpolant: kernel φ plus a linear polynomial, optional smoothing λ on
   the diagonal. Predict foreground where the interpolant is ≥ 0. Runs once
   per subject.
2. **`ssm` (population shape prior):** principal component analysis of
   training subjects' 3D signed distance maps in the fixed crop frame (below),
   with no further alignment. Fitting optimizes K mode coefficients and a 3D
   translation by Adam (300 steps, learning rate 0.05, zero initialization)
   to minimize the mean binary cross-entropy between `sigmoid(SDF / 2 mm)` and
   the observed labels, plus λ · Σ b_k², with b_k in units of the mode's SD. Predict foreground where the
   fitted SDF is ≥ 0. Deterministic; runs once per subject.
3. **`unet` (learned feed-forward completion):** a 3D U-Net (four resolution
   levels, 16 base channels, instance normalization) taking two channels in
   the crop frame (observed label value, observation mask) and predicting
   occupancy logits. Trained with binary cross-entropy plus soft Dice on full
   training labels, AdamW, mixed precision, batch size 1, on inputs built from
   the two protocol layouts with plane positions jittered by up to ±6 voxels
   per plane during training only. Evaluation uses the exact protocol planes.
4. **`gp` (non-learned, with uncertainty):** Gaussian-process regression on
   the same constraints and centres as `rbf`, with a Matérn-5/2 kernel of
   length ℓ, a constant prior mean m (mm), signal variance equal to the mean
   squared residual about m, and noise SD 0.5 mm. Predict foreground where the
   posterior mean is ≥ 0. Far from the sections the mean reverts to m, so
   m < 0 encodes "background unless observed". The posterior SD (on the
   stride-2 grid) is reported descriptively: AUROC of −|mean|/SD for
   misclassified unobserved voxels, the share of misclassified voxels within
   1.96 SD of zero, and median SD. Kernel values are computed in float32 and
   summed in float64 (checked against float64 on one fold-0 subject: max
   difference 0.0007 mm, 1 of 296,140 voxels changed sign). Runs once per
   subject.
5. **`prior_inr` (population-initialised INR):** fit `ssm` with its frozen
   fold-0 configuration (K = 20, λ = 0.001) to the sections; distil that
   prior occupancy into the v1 INR architecture (1,000 Adam steps, lr 1e-3,
   batch 4,096 uniformly sampled grid voxels); then fine-tune on the observed
   sections with v1's step budget (1,000 steps), batch size, and per-seed
   minibatch sequence, at fine-tuning learning rate η, adding β times the BCE
   to the prior occupancy on 4,096 uniformly sampled unobserved voxels per
   step. Seeds 17, 23, 42 as v1 INR. Uses the same cross-fitted `ssm`
   artifact as `ssm`.

**Crop frame.** `ssm` and `unet` operate on a 128 × 128 × 160 voxel crop
centred on the array centre `(shape - 1) / 2`, computed from the array shape
only and padded with background where the array is smaller. Predictions are
pasted back into the native grid; voxels outside the crop are predicted
background. All metrics are computed on the native grid exactly as in v1, so
any target foreground outside the crop counts as error.

## Tuning on fold 0 (bounded grids)

Each method selects one configuration, shared by both layouts, from the grid
below. The selection score is v1's: unobserved-voxel Dice, averaged over seeds
within subject, then over the two layouts, then over the 22 fold-0 subjects.
Ties within 0.002 go to the first configuration listed.

| Method | Grid | Configurations |
| --- | --- | --- |
| `rbf` | φ(r) ∈ {r, r³} × λ ∈ {0, 0.01} | 4 |
| `ssm` | K ∈ {5, 10, 20, 40} × λ ∈ {0.001, 0.01, 0.1} | 12 |
| `unet` | learning rate ∈ {3e-4, 1e-3} × training iterations ∈ {4,000, 8,000} | 4 |
| `gp` | ℓ ∈ {10, 20, 40} mm × m ∈ {0, −10} mm | 6 |
| `prior_inr` | β ∈ {0, 0.1, 1} × η ∈ {1e-4, 1e-3} | 6 |

Grid order for the tie rule is as listed, last factor fastest (for `gp`:
ℓ = 10/m = 0, ℓ = 10/m = −10, ℓ = 20/m = 0, …).

For fold-0 tuning, population models were trained on folds 1–4. Although no
fold-1–4 predictions were scored during tuning, their labels influenced model
selection through the learned population priors. The v2 results on these folds
are therefore exploratory, not independent confirmation.

The number of U-Net training seeds is fixed before freezing from measured
training time: three seeds (17, 23, 42) if one training run takes at most 45
minutes on the project GPU, otherwise seed 17 only.

## Frozen settings and final held-out test evaluation

The fold-0 selections are fixed before test access:

- `rbf`: φ(r) = r³, λ = 0.01.
- `ssm`: K = 20, λ = 0.001.
- `unet`: learning rate 3e-4, 4,000 iterations, training seeds 17, 23, 42.
- `gp`: Matérn-5/2 length ℓ = 20 mm, prior mean m = −10 mm.
- `prior_inr`: β = 0, fine-tuning learning rate η = 1e-4; uses the selected
  `ssm` configuration.
- v1 comparators: INR settings from v1; Gaussian occupancy uses the selected
  uniform initialization and learning rate 0.001; interpolation is unchanged.

Train each population model once on all five development folds (108 subjects)
using scan-1 ED labels only. Save its training-subject list and verify it is
disjoint from the frozen 26-subject test set. Evaluate all methods on exactly
those test subjects under both layouts. The runner rejects evaluation of any
subject in a population model's training list and requires explicit
`--allow-test-evaluation` to select the test subset. Save predictions only for
MITEA_004 and MITEA_017 (the first two test IDs in manifest order, selected
before loading any test labels). Verify that v2 and v1 comparators have
identical observation fingerprints for each subject and layout.

This is a one-time held-out evaluation (n = 26), not a cross-validation
estimate. Do not pool it with the separate v1 86-subject confirmatory estimate.

## Statistical analysis

- **Primary comparisons (10):** `rbf`, `gp`, `ssm`, `unet`, and `prior_inr`
  each against v1 INR, for each layout, on unobserved-voxel Dice. Two-sided
  paired Wilcoxon signed-rank tests on subject-level values, Holm correction
  over these ten.
- **Secondary (descriptive, with bootstrap CIs, no Holm family):** `rbf` vs the
  v1 layout-specific interpolation; `gp` vs `rbf`; `unet` vs `ssm`;
  `prior_inr` vs `ssm`; Gaussian occupancy vs INR and interpolation; GP
  uncertainty summaries; all secondary metrics.
- Bootstrap: 10,000 replicates, RNG seed 2027, as in v1.
- Largest-component sensitivity and pathology subgroups as in v1.

## Deviation log

### 2026-10-20 — Development history before fold-0 tuning (disclosure)

The tuning grids above were written before any fold-0 tuning run. Before
that, the implementations were smoke-checked on one fold-0 subject
(MITEA_007): `rbf` with both kernels and `ssm` with three (K, λ) settings,
and a 40-iteration U-Net timing run on six fold-1 training subjects. While
implementing `ssm`, the prior scale was changed from the first draft
(λ ∈ {0.1, 1, 10} on Σ(b/σ)²) to mean BCE + λ·Σb² with λ ∈ {0.001, 0.01,
0.1}, because the first draft's scale would let the prior dominate a
per-voxel mean loss; this was decided from the loss definition, not from
results. No fold-1–4 subject has been evaluated by any v2 method.

### 2026-10-20 — Fold-0 selections (rule applied as written)

Fold-0 tuning ran 2026-10-20 (rbf/ssm 4 + 12 configurations, unet 4
configurations, both layouts, 22 subjects each); 0 failed runs. Selection files
are in `runs/protocol_v2_selection/fold0_{rbf,ssm,unet}.json`. Layout-averaged
unobserved Dice of the selected configuration (orthogonal / axial):

- `rbf`: φ = r³, λ = 0.01 — 0.9675 / 0.8947, average 0.9311 (best outright).
- `ssm`: K = 20, λ = 0.001 — 0.9621 / 0.9628, average 0.9624. K = 40 scored
  0.9631, within the 0.002 tie margin, so the earlier K = 20 is selected.
  λ = 0.001 is the smallest value in the grid and scores rose as λ fell; the
  grid is not extended (this would be a post-hoc change), and the boundary
  optimum is reported.
- `unet`: learning rate 3e-4, 4,000 iterations — 0.9705 / 0.9701, average
  0.9703. All four configurations lay within 0.0013 of each other, so the
  first-listed one is selected by the tie rule.

U-Net seeds: the measured 8,000-iteration training run took 31.5 min (and
42.7 min while sharing the GPU with the rbf/ssm grid), so a 4,000-iteration
run is well under 45 min and the confirmatory analysis uses three seeds
(17, 23, 42).

Context, computed after tuning and not used for selection: the folds-1–4 mean
shape (thresholded mean signed distance, no sections) gives whole-volume Dice
0.814 (minimum 0.669) on fold 0, so MITEA labels sit at a consistent position
and size in the array. Proposed (before freezing) as a descriptive reference
row in the confirmatory report.

### 2026-10-21 — Two methods added after the first three were tuned (disclosure)

At the authors' request, `gp` and `prior_inr` were added to this draft on
2026-10-21, after the fold-0 results and selections of `rbf`, `ssm`, and
`unet` above had been seen. Both were among the options listed when v2 was
scoped but were not in the first draft. What was known when their grids were
written: the fold-0 numbers above (in particular, `rbf` is weaker on the
parallel layout and `ssm` scores similarly on both layouts), and a smoke
check on MITEA_007 (fold 0, parallel layout): `gp` ℓ = 20, m = −10 gave
unobserved Dice 0.876; `prior_inr` β = 0.1, η = 1e-3, seed 17 gave 0.964; on
the orthogonal layout `gp` gave 0.888 (ℓ = 10, m = −10), 0.930 (ℓ = 10, m = 0),
and 0.954 (ℓ = 20, m = −10).
`prior_inr` reuses the `ssm` selection rather than retuning it. Before the
smoke check, `gp` was sped up (float32 kernel values) after a 145 s per
subject first version; the accuracy check is reported in Methods. The
primary family grows from 6 to 10 tests; Holm correction covers all 10.
No fold-1–4 subject has been evaluated by any v2 method.

### 2026-10-21 — Fold-0 selection for `gp` (rule applied as written)

Six configurations × two layouts × 22 subjects, 0 failed runs
(`runs/protocol_v2_selection/fold0_gp.json`). Selected ℓ = 20 mm, m = −10 mm:
unobserved Dice 0.9596 orthogonal / 0.8987 axial, average 0.9291 (best
outright; next ℓ = 40, m = −10 at 0.9125). The negative prior mean mattered
most on the parallel layout (ℓ = 20: 0.683 with m = 0 vs 0.899 with m = −10).
Descriptive uncertainty for the selected configuration (median over
subjects): error AUROC 0.994 orthogonal / 0.914 axial; share of
misclassified unobserved voxels within 1.96 SD of zero 1.000 / 0.986.
`prior_inr` tuning started after `gp` finished.

### 2026-10-21 — Fold-0 selection for `prior_inr` (rule applied as written)

Six configurations × two layouts × 22 subjects × 3 seeds, 0 failed runs
(`runs/protocol_v2_selection/fold0_prior_inr.json`). Selected β = 0,
η = 1e-4: unobserved Dice 0.9737 orthogonal / 0.9690 axial, average 0.9713.
β = 0.1, η = 1e-4 scored 0.9721, within the 0.002 tie margin, so the earlier
β = 0 is selected. All six configurations lay between 0.9665 and 0.9721.
Median 26-connected component count was 1 in every configuration; median
fit time 8.7 s (β = 0) including the shape-model fit. For context, `ssm`
alone (same artifact and settings) scored 0.9621 / 0.9628 on fold 0.

### 2026-10-22 — Final held-out evaluation design (frozen before test access)

The authors authorized one-time use of the 26-subject test set. Review of the
original cross-fitting proposal found that folds 1–4 contributed labels to the
population models used in fold-0 tuning; those folds are consequently not
described as independent v2 confirmation. Selected settings above are frozen
without consulting test labels or scores. Final population models will be
trained on folds 0–4 only, with the test set excluded. The test set will be
evaluated once; failures and deviations will be reported. The original v1
protocol and its 86-subject estimate remain unchanged.

### 2026-10-22 — Held-out test results (one-time evaluation, as frozen)

Final population models were trained on folds 0–4 (108 subjects; SSM and
three U-Net seeds) and evaluated once on the 26 test subjects with
`scripts/run_protocol_v2_test.sh`. 416/416 runs succeeded in each layout,
0 failures, 0 empty predictions; every subject has a single supervision
fingerprint per layout, so all methods saw identical observations. Report:
`runs/protocol_v2_test_20261022_report.{json,tex}`.

Unobserved-voxel Dice, mean ± SD over subjects (orthogonal / axial):

| Method | Orthogonal | Axial |
| --- | --- | --- |
| `unet` | 0.968 ± 0.007 | 0.968 ± 0.008 |
| `prior_inr` | 0.969 ± 0.006 | 0.967 ± 0.007 |
| `rbf` | 0.965 ± 0.007 | 0.899 ± 0.019 |
| `gp` | 0.956 ± 0.010 | 0.897 ± 0.014 |
| `ssm` | 0.954 ± 0.011 | 0.956 ± 0.010 |
| v1 INR | 0.933 ± 0.008 | 0.515 ± 0.036 |
| v1 interpolation | 0.923 ± 0.012 | 0.552 ± 0.029 |
| Gaussian occupancy | 0.902 ± 0.012 | 0.814 ± 0.034 |

Primary family: all 10 comparisons favor the v2 method over INR (Holm
p = 3.0e-7 each). Orthogonal mean gains +0.021 (`ssm`) to +0.036
(`prior_inr`), with 25–26/26 subjects improved; axial gains +0.38 (`gp`) to
+0.45 (`unet`, `prior_inr`), 26/26 improved. Secondary (descriptive): `gp`
did not beat `rbf` (orthogonal −0.009, 0/26 higher; axial −0.002, CI
−0.010 to +0.005); `unet` and `prior_inr` exceed `ssm` by +0.011 to +0.015;
Gaussian occupancy is below INR on orthogonal (−0.032, 0/26) but above it on
axial (+0.299). This 26-subject estimate is not pooled with the v1
86-subject estimate.
