# MITEA sensitivity analyses v1: results

These results follow the frozen [protocol](mitea_sensitivity_v1.md). They are
development-cohort sensitivity analyses on folds 1–4 (86 subjects) using seed
17, and the test set was not used.

The run completed 2026-10-29: 104/104 units, 0 failed runs, 0 empty
predictions, and no observation-fingerprint conflicts. No perturbation emptied
a section. The full tables are in the git-ignored
`runs/mitea_sensitivity_v1/analysis/` (`results.md`, `summary.json`),
produced by `scripts/summarize_sensitivity.py`.

Dice is unobserved-voxel Dice. Each difference is given as mean [95% bootstrap
CI] with the number of subjects higher. *Material* means the CI excludes 0
and |mean| ≥ 0.01.

## A. Imperfect sections

**Primary (clean → random 2 mm), change in Dice:**

| Method | Orthogonal | Axial |
| --- | --- | --- |
| `ssm` | −0.006 [−0.007, −0.005] | −0.007 [−0.008, −0.005] |
| `unet` | −0.026 [−0.028, −0.025] | −0.023 [−0.024, −0.022] |
| `prior_inr` | −0.031 [−0.033, −0.030] | −0.024 [−0.025, −0.022] |
| `rbf` | −0.029 [−0.030, −0.027] | −0.017 [−0.022, −0.014] |
| `gp` | −0.029 [−0.030, −0.028] | −0.018 [−0.020, −0.017] |
| `inr` | −0.037 [−0.040, −0.035] | +0.002 [−0.001, +0.006] |
| `interpolation` | −0.025 [−0.027, −0.023] | −0.008 [−0.010, −0.007] |

- **Every method except `ssm` loses a material 0.02–0.04 Dice at 2 mm noise
  and 0.05–0.10 at 4 mm.** `ssm` loses 0.006 (not material) and 0.019. The
  exceptions are axial `inr` and `interpolation`, which already fail on clean
  sections.
- **Noise changes the ranking.** On clean sections `prior_inr` ranks first. At
  2 mm `ssm` ranks first on both layouts (0.949 orthogonal and 0.948 axial,
  against 0.940/0.941 for `prior_inr`). At 4 mm it leads by about 0.03. The
  low-dimensional shape model filters tracing noise. Methods that fit the
  sections closely (`prior_inr`, `rbf`, `gp`) or were trained on clean inputs
  (`unet`) pass the noise through.
- **No method corrects a systematic tracing bias.** With a 2 mm inward or
  outward offset, Dice drops by 0.02–0.07 and the mean signed volume error
  shifts to about −45 mL or +45 mL for every method that works. On clean
  sections the error is within ±6 mL for all methods except INR on the
  orthogonal layout (−13.5 mL).

## B. Prior composition and training-set size

**B1 (primary).** On the 33 pathology subjects, the healthy-only prior minus
the size-matched mixed prior:

- `ssm`: −0.006 (orthogonal), −0.007 (axial)
- `unet`: −0.001, −0.005
- `prior_inr`: −0.001, −0.006

None is material. The unfitted mean shape alone is materially worse (−0.030).
Once a prior is fitted to the sections, training it only on healthy hearts
costs pathological hearts less than 0.01 Dice.

**B2.** The size of the training set matters little:

- `unet` and orthogonal `prior_inr` are within 0.01 of the full-pool score
  with 10 training shapes.
- `ssm` and axial `prior_inr` need 25.

At 10 shapes, `unet` scores 0.963/0.955, against 0.967/0.962 with the full
pool of 86–87.

## C. Plane count

**Mean shape alone.** With no sections, the population mean shape scores
0.799 whole-volume Dice. The three protocol planes raise the best methods to
about 0.97.

**One plane.** The priors (`ssm` and `prior_inr`, about 0.89) beat `gp`
(about 0.82), `inr` and `interpolation` (about 0.45–0.47).

**Orthogonal family.** `rbf` comes within 0.01 of `prior_inr` at three planes
(−0.005). Two orthogonal planes already give `prior_inr` 0.965 and `ssm`
0.957.

**Parallel family.** No per-subject method comes within 0.01 of `prior_inr`
even with six sections:

| Method | Gap to `prior_inr` at 6 sections |
| --- | --- |
| `rbf` | −0.026 |
| `gp` | −0.032 |
| `interpolation` | −0.108 |
| `inr` | −0.239 |

`prior_inr` itself plateaus at 0.967 from four sections on.

**U-Net at other plane counts.** The `unet` was trained on three-plane inputs
and degrades at other counts:

- one plane: 0.814 orthogonal and 0.821 axial, against about 0.89 for `ssm`;
- six sections: 0.954 against 0.962 at three.

This mismatch is a practical limitation of feed-forward completion.

## Interpretation for the manuscript

The v2 conclusion, that population priors beat per-subject fitting, holds
under all three stresses. Two qualifications belong in the paper:

1. With realistic tracing noise, the constrained SSM is the most robust
   method, so the clean-section ranking of `prior_inr` and `unet` should not
   be read as robustness.
2. Systematic tracing bias propagates almost one-to-one into volume error for
   every method.
