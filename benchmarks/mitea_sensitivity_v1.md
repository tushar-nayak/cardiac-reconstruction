# MITEA Sensitivity Analyses v1

**Status: frozen (2026-10-28) before any run on the evaluation cohort.**
Smoke checks used only the fold-0 subject MITEA_007 (see the deviation log).
No setting may change after evaluation-cohort outputs are inspected.

## Purpose

The protocol v2 held-out evaluation found that population priors (`unet`,
`prior_inr`, `ssm`) complete unobserved anatomy better than per-subject
fitting, most strongly on the parallel layout. Three questions qualify that
result and are not answered by it:

- **A. Imperfect sections.** Every input so far is an exact label section.
  How do the methods degrade when the observed sections carry tracing error?
- **B. Prior composition and size.** Does a prior trained only on healthy
  hearts do worse on pathological ones, and how many training shapes do the
  priors need?
- **C. Plane count.** How does completion change with the number of sections,
  and what do the sections add over the population mean shape alone?

These are sensitivity analyses on the development cohort. They were designed
after the v1 confirmatory (folds 1–4) and v2 test results were known. They
report estimates with confidence intervals and no hypothesis tests, and they do
not replace or update either earlier estimate.

## Common design

- **Cohort.** Frozen manifest `benchmarks/splits/mitea_cohort_v1.json`.
  Evaluate development folds 1–4 (86 subjects). Fold 0 is excluded because the
  v2 settings were selected on it. The 26-subject test set is not loaded: no
  command passes `--allow-test-evaluation`.
- **Target, layouts, metrics.** As in v1/v2: scan 1, ED, label union, on the
  1 mm isotropic arrays. Layouts are orthogonal central planes and parallel
  axis-0 planes at 0.375/0.5/0.625 unless a study changes them. Threshold 0.5.
  Primary metric: unobserved-voxel Dice. Secondary: whole Dice, HD95, ASSD,
  signed and absolute volume error, and 26-connected component count.
- **Methods.** The frozen v2 settings, with no retuning: `inr`,
  `interpolation`, `rbf`, `ssm`, `unet`, `gp`, `prior_inr`, plus the
  `mean_shape` reference below. Gaussian occupancy is excluded (secondary in
  v1/v2, and costly). GP uncertainty summaries are not computed.
- **Seeds.** Seed 17 only, for `inr`, `prior_inr`, and U-Net training. On
  fold 0, the median within-subject SD of unobserved Dice across seeds 17, 23
  and 42 was 0.012 (axial) and 0.005 (orthogonal) for v1 `inr`, and below 0.001
  for `prior_inr`. Over 86 subjects this adds about 0.001 to the standard error
  of a cohort mean.
- **Cross-fitted reference priors.** For evaluation fold k ∈ {1, 2, 3, 4},
  build `ssm` and train `unet` (seed 17) on the other four development folds
  (86–87 subjects, including fold 0) using scan-1 ED labels. The runner rejects
  any evaluated subject that appears in a model's training list.
- **`mean_shape` reference.** The training SSM's mean signed-distance map,
  thresholded at 0 in the crop frame and pasted into the native grid, with no
  fit to the sections. It is scored on the same unobserved voxels as the other
  methods, so `method − mean_shape` measures what the sections add. With zero
  sections its unobserved Dice equals whole-volume Dice.
- **Unit of inference and statistics.** The subject. For each method and
  condition, report mean, SD, median [IQR], and a 95% percentile bootstrap CI
  over subjects (10,000 replicates, RNG seed 2027). Paired differences are
  taken within subject, with bootstrap CIs and the number of subjects improved.
  No p-values. A difference is called material only if its CI excludes 0 and
  |mean| ≥ 0.01 Dice. CIs reflect subject sampling, not the training randomness
  of population models.
- **Integrity checks.** Zero failed runs and zero empty predictions are
  expected; any that occur are reported. Within each subject, layout, and
  condition, all methods must share one observation fingerprint.

## Study A: imperfect sections

**Perturbation.** Only the observed sections are perturbed; evaluation always
uses the clean target. For each observed 2D section S (foreground of the label
union), compute the signed distance d (mm) to its boundary, placed midway
between foreground and background pixel centres: d = EDT(S) − ½ inside and
d = −(EDT(¬S) − ½) outside. The perturbed section is

  S′ = {d + c + σ·g ≥ 0},

where c is a systematic offset (tracing consistently outside the true border,
c > 0, or inside it, c < 0) and g is a smooth unit-variance random field:
i.i.d. standard normal noise on the section grid, Gaussian-filtered with SD
5 mm, then divided by its SD over the section. g is drawn once per subject,
layout, and plane from
`numpy.random.SeedSequence([2027, subject number, layout index, plane index])`
(orthogonal = 0, parallel = 1) and reused across σ levels (common random
numbers).

- Sections with no foreground or no background are left unchanged.
- Where orthogonal planes intersect, the label from the plane with the lowest
  array axis is used.
- No smoothing, component filtering, or hole filling is applied to S′.

**Conditions** (both layouts, 86 subjects, all methods):

| Condition | c (mm) | σ (mm) |
| --- | --- | --- |
| clean | 0 | 0 |
| random 2 | 0 | 2 |
| random 4 | 0 | 4 |
| inside 2 | −2 | 0 |
| outside 2 | +2 | 0 |

The clean condition reproduces S exactly and is also the reference for
Studies B and C. `mean_shape` ignores the sections, so it runs only in the
clean condition.

**Primary estimand.** For each method and layout, the mean paired change in
unobserved Dice from clean to random 2.

**Secondary.**
- The random 4 condition.
- Signed volume error under the ±2 mm offsets, to show how a tracing bias
  propagates.
- Whether the clean ordering of methods holds under perturbation.
- All secondary metrics.
- The number of sections the perturbation emptied.

## Study B: prior composition and training-set size

Only the population methods are refitted: `ssm`, `prior_inr`, `unet`, and
`mean_shape`. Per-subject methods do not depend on the prior; their Study A
clean results are shown for context.

**B1: composition (primary for Study B).** Evaluate the 33 fold-1–4 subjects
with a pathology label: hypertrophy 8, amyloidosis 8, aortic regurgitation 6,
hypertrophic cardiomyopathy 5, dilated cardiomyopathy 4, transplant 2. Compare
two priors of equal size:

- **H:** trained on all 66 healthy development subjects. No evaluated subject
  is among them.
- **M66:** for each fold k, 66 subjects sampled without replacement from that
  fold's training pool, keeping the pool's healthy:pathology ratio after
  rounding (25–27 pathology subjects), with RNG seed 2027 + k.

The primary estimand is, for each method and layout, the mean paired
difference H − M66 in unobserved Dice. Per-category results are descriptive
only (2–8 subjects each). The cross-fitted reference priors (Study A clean)
are shown for context.

**B2: training-set size (secondary).** For each fold k, shuffle the
pathology and healthy subjects of its training pool separately (RNG seed
3027 + k) and take nested subsets of size n ∈ {10, 25, 50}, each containing
round(n·p) pathology subjects, where p is the pool's pathology fraction. On
each subset, train `ssm` with K = min(20, n − 1) and `unet` (seed 17, unchanged
recipe including 4,000 iterations), then evaluate all 86 subjects on both
layouts. The full pool (86–87 subjects) is the Study A clean result. Report
unobserved Dice against n for each method and layout, and the smallest n whose
mean is within 0.01 of the full-pool mean.

## Study C: plane count

**Layouts.**
- **Orthogonal family:** 1 plane (axis 0) and 2 planes (axes 0 and 1), which
  are nested prefixes of the v2 orthogonal layout.
- **Parallel family:** n ∈ {1, 2, 4, 6} sections at a fixed spacing of 1/8 of
  the axis-0 depth, centred at 0.5: fractions 0.5 + (i − (n − 1)/2)/8 for
  i = 0, …, n − 1. For n = 3 this rule gives the v2 layout. For n = 6 the
  sections span 0.1875–0.8125, which covers the median foreground extent
  (0.319–0.750, from v1 Amendment 1).
- **Three planes**, in either family, are the Study A clean condition.
- **Zero planes:** `mean_shape` only, scored with whole-volume metrics.

**Methods.** All methods run at every count, with these exceptions fixed in
advance:
- `rbf` is not run with one plane, because its linear polynomial term is
  undetermined when all constraints are coplanar.
- `unet` was trained on three-plane inputs only. At other counts it is
  reported as an acquisition-mismatch result, not as the method's attainable
  accuracy.
- `interpolation` uses the frozen layout-specific baselines:
  section-extrusion intersection, and axial signed-distance interpolation
  (which extrudes a single section when n = 1).

**Primary estimand.** In the parallel family: for each method, mean unobserved
Dice by count, and the paired gap of each per-subject method (`inr`, `rbf`,
`gp`, `interpolation`) to `prior_inr` at the same count.

**Secondary.** The orthogonal family; gaps to `mean_shape`; and the smallest
count at which each per-subject method comes within 0.01 of `prior_inr`.

## Compute and order

Population models: 21 U-Net trainings (4 reference, 1 H, 4 M66, 12 B2) at
about 28 minutes each, about 10 h in total, plus 21 SSM builds at about
1.5 minutes each. Evaluation estimates come from single-process smoke timings
on MITEA_007: about 2 minutes per subject for both layouts in Study A, giving
about 14 h (A), 8 h (C), and 6 h (B). The total is about 39 GPU-hours.

Run order: train the reference priors → A → C → B1 → B2. Training and
evaluation run one process at a time.

## Reproducibility

Outputs go under the git-ignored `runs/mitea_sensitivity_v1/`, with one
directory per study, fold, and condition. `scripts/run_sensitivity_v1.sh`
runs every step. It skips finished units and refuses to overwrite a partial
one. Subject lists for H, M66, and the B2 subsets are written to JSON by
`scripts/make_sensitivity_subsets.py` before any training and recorded in each
model artifact. `scripts/summarize_sensitivity.py` computes the estimands
above. Results are summarized in `benchmarks/mitea_sensitivity_v1_results.md`.

## Deviation log

### 2026-10-28 — Information available when drafting (disclosure)

This protocol was drafted after the v1 confirmatory (folds 1–4) and v2 test
results were known. Design facts consulted:

- development pathology counts per fold;
- fold-0 seed variability (above);
- per-method runtimes from the v2 test run (timing only);
- the v1 Amendment 1 extent statistics.

No method has been run under any perturbation, prior variant, or plane count
described here.

### 2026-10-28 — Implementation and smoke checks before freezing

Implementation:

- section perturbation in `cardiac_common/perturb.py`, applied by
  `scripts/benchmark_sparse.py` (`--section-offset-mm`,
  `--section-noise-sd-mm`, `--section-noise-filter-mm`), with SciPy's default
  reflecting boundary for the Gaussian filter;
- the `mean_shape` reference;
- `--no-gp-uncertainty`;
- `--train-subjects-file` for `scripts/train_ssm.py` and
  `scripts/train_unet.py`.

Without a perturbation, the runner's inputs are unchanged.

Smoke checks used MITEA_007 only, with the v2 fold-0 tuning models (trained on
folds 1–4). Outputs are in `runs/mitea_sensitivity_v1_smoke_20261028/`.

- The clean path reproduced the v2 fold-0 tuning scores (unobserved Dice to
  four decimals) for `inr`, `rbf`, `ssm`, `unet`, `gp`, and `prior_inr` on
  both layouts.
- These configurations completed without failures, each with one observation
  fingerprint per run:
  - random 2 mm (orthogonal) and inside 2 mm (parallel);
  - one and six parallel sections, and one orthogonal plane;
  - a 10-subject SSM (9 modes) and a short U-Net run trained from a subject
    list.
- Two evaluation processes run at once filled the 8 GB GPU (7.97 GB) and
  slowed `gp` about fivefold and `prior_inr` about sevenfold. Training and
  evaluation therefore run one process at a time.

The subject lists were generated and verified by
`scripts/make_sensitivity_subsets.py`. No evaluation-cohort subject has been
run.
