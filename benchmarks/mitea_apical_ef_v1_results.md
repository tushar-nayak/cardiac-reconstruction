# MITEA apical-view EF study v1: results

These results follow the frozen [protocol](mitea_apical_ef_v1.md). The study
used development folds 1–4 (86 subjects), scan 1 at end-diastole (ED) and
end-systole (ES), with seed 17. The test set was not used. This is a
development-cohort study, so it gives estimates with 95% subject-bootstrap
CIs and no p-values.

The run took place 2026-11-02 16:06 to 2026-11-03 01:00.

- **Integrity.** 4/4 units completed, with 8,084/8,084 rows, 0 failed runs,
  and 0 empty predictions.
- **Observations.** Within each subject, phase, layout, and condition, every
  method read the same observation fingerprint.
- **Coverage.** Every comparison includes all 86 subjects.
- **Outputs.** The full tables are in the git-ignored
  `runs/mitea_apical_ef_v1/analysis/` (`results.md`, `summary.json`), produced
  by `scripts/summarize_apical_ef.py`.

## Conventions

- **Units.** EF errors are in percentage points (pp).
- **Differences.** "Δ vs Simpson" is the paired difference in absolute error,
  method minus biplane Simpson, as mean [95% CI] with the number of subjects
  the method improves. Negative favours the method.
- **Material.** *Material* (bold) means the CI excludes 0 and the magnitude is
  at least 1 pp for EF (5 mL for volumes).
- **Simpson and pose.** Simpson does not depend on the pose condition, so its
  rows are the same under known and assumed pose.

## Primary: biplane views, known pose

| Misplacement | Method | EF MAE [95% CI] | EF bias | Δ vs Simpson (EF) | Better |
| --- | --- | --- | ---: | --- | ---: |
| none | `simpson` | 0.68 [0.57, 0.80] | +0.53 | – | – |
| | `ssm` | 0.57 [0.48, 0.66] | −0.22 | −0.11 [−0.27, +0.03] | 54/86 |
| | `unet` | 0.55 [0.46, 0.64] | +0.12 | −0.14 [−0.25, −0.03] | 55/86 |
| | `prior_inr` | 0.77 [0.63, 0.94] | +0.32 | +0.09 [−0.08, +0.26] | 39/86 |
| | `rbf` | 1.11 [0.99, 1.22] | +1.10 | +0.42 [+0.32, +0.52] | 17/86 |
| | `inr` | 3.41 [2.81, 4.06] | −0.74 | **+2.73 [+2.12, +3.39]** | 8/86 |
| 5° / 3 mm | `simpson` | 2.68 [2.30, 3.10] | +2.15 | – | – |
| | `ssm` | 0.82 [0.67, 0.98] | −0.13 | **−1.86 [−2.25, −1.49]** | 74/86 |
| | `unet` | 0.85 [0.69, 1.04] | +0.10 | **−1.83 [−2.22, −1.45]** | 75/86 |
| | `prior_inr` | 1.11 [0.93, 1.30] | +0.21 | **−1.57 [−1.97, −1.18]** | 71/86 |
| | `rbf` | 1.21 [1.04, 1.40] | +0.57 | **−1.47 [−1.89, −1.06]** | 67/86 |
| | `inr` | 4.15 [3.49, 4.91] | +0.50 | **+1.47 [+0.70, +2.28]** | 30/86 |
| 10° / 5 mm | `simpson` | 5.98 [5.17, 6.82] | +3.84 | – | – |
| | `ssm` | 1.08 [0.90, 1.27] | −0.12 | **−4.90 [−5.72, −4.09]** | 76/86 |
| | `prior_inr` | 1.44 [1.23, 1.67] | +0.09 | **−4.53 [−5.35, −3.71]** | 75/86 |
| | `unet` | 1.77 [1.43, 2.14] | +0.32 | **−4.21 [−5.02, −3.38]** | 74/86 |
| | `rbf` | 2.50 [2.07, 2.96] | +0.55 | **−3.48 [−4.37, −2.60]** | 65/86 |
| | `inr` | 4.66 [3.92, 5.48] | +0.98 | **−1.31 [−2.33, −0.28]** | 52/86 |

**Findings.**

- **Well-placed views.** With perfectly placed views, Simpson is already
  within 0.7 pp, and no reconstruction is materially better. `ssm` and `unet`
  are within 0.15 pp of it. `rbf` is slightly worse (+0.4 pp, not material)
  because it overestimates EDV by 5.5 mL. Direct `inr` is materially worse
  (+2.7 pp).
- **Misplaced views.** Once the views are misplaced and their pose is known,
  every prior-based method is materially better than Simpson at both levels.
  - *Error growth.* Simpson's EF error grows from 0.7 to 2.7 to 6.0 pp.
  - *Methods.* `ssm` stays at 0.6 to 0.8 to 1.1 pp. `prior_inr` and `unet`
    stay below 1.8 pp. At 10° / 5 mm, the 95% limits of agreement for `ssm`
    are [−2.9, +2.6] pp, against [−8.1, +15.8] pp for Simpson.
  - *Breadth.* The gain is broad: 74–76 of 86 subjects improve.
- **Volumes.** The gain comes from the volumes.
  - *Simpson.* Misplaced views make Simpson underestimate both volumes. At
    10° / 5 mm its EDV bias is −22.7 mL and its ESV bias −14.3 mL, with MAEs
    of 22.7 and 14.3 mL.
  - *Shape priors.* `ssm` and `prior_inr` keep EDV MAE at 3.2–3.7 mL and ESV
    MAE at 1.8–2.1 mL.
  - *Paired.* The paired EDV improvement over Simpson is −19.0 mL
    [−21.5, −16.6] for `ssm` and −19.5 mL [−22.1, −17.1] for `prior_inr`
    (84/86 subjects). Both are material.
- **No shape prior.** `rbf` helps, but less. Its EDV MAE is 11.7 mL at the
  severe level. Direct `inr` is worse than Simpson unless the views are
  severely misplaced.
- **Ranking.** The best method depends on the misplacement level. `unet` is
  marginally best with perfect views, and `ssm` is best at both misplacement
  levels. This matches the sensitivity analyses: the low-dimensional shape
  model is the most robust to imperfect input.

## Secondary

### Assumed pose (biplane)

Here the misplaced images are placed on the intended planes, so the true pose
is unknown.

| Misplacement | Method | EF MAE | Δ vs Simpson (EF) | Better | EDV bias (mL) |
| --- | --- | ---: | --- | ---: | ---: |
| 5° / 3 mm | `simpson` | 2.68 | – | – | −9.0 |
| | `ssm` | 2.00 | −0.68 [−0.88, −0.48] | 65/86 | −7.9 |
| | `unet` | 2.10 | −0.58 [−0.75, −0.41] | 69/86 | −6.1 |
| | `prior_inr` | 2.31 | −0.37 [−0.60, −0.14] | 52/86 | −5.8 |
| | `rbf` | 2.74 | +0.06 [−0.10, +0.21] | 38/86 | +0.1 |
| | `inr` | 3.99 | **+1.31 [+0.54, +2.15]** | 38/86 | +12.0 |
| 10° / 5 mm | `simpson` | 5.98 | – | – | −22.7 |
| | `unet` | 5.02 | −0.96 [−1.31, −0.61] | 62/86 | −18.5 |
| | `prior_inr` | 5.10 | −0.87 [−1.26, −0.49] | 55/86 | −18.0 |
| | `ssm` | 5.32 | −0.66 [−1.03, −0.30] | 52/86 | −20.4 |
| | `rbf` | 5.30 | −0.68 [−0.98, −0.38] | 57/86 | −13.1 |
| | `inr` | 5.71 | −0.27 [−1.18, +0.69] | 49/86 | −2.3 |

**Findings.**

- **Without the true pose, most of the benefit disappears.**
  - *EF.* The prior-based methods stay slightly better than Simpson, with CIs
    that exclude 0, but by 0.4–1.0 pp. That is below the 1 pp materiality
    threshold.
  - *Volumes.* They inherit Simpson's volume underestimate: an EDV bias of
    −18 to −20 mL at the severe level, against −22.7 mL for Simpson.
- **Why.** A misplaced image placed on the intended plane shows a cavity
  section that is genuinely smaller, and a shape prior cannot tell that apart
  from a smaller heart.
- **Interpretation.** The large known-pose gain therefore depends on knowing
  where the views actually are, for example from probe tracking, 3D
  acquisition, or registration of the views to each other.

### Triplane layout (0°, 60°, 120°)

Each triplane reconstruction is compared with biplane Simpson at the same
misplacement level.

- **No misplacement.** `rbf` (0.44 pp) and `unet` (0.43 pp) are the most
  accurate EF estimates in the study. Their EDV and ESV MAEs are 1.0–2.1 mL
  and 0.6–0.9 mL. They are better than Simpson by 0.25 pp, which is not
  material.
- **10° / 5 mm, known pose.** All prior-based methods and `rbf` are materially
  better than Simpson, by −4.4 to −4.9 pp in EF and −18 to −20 mL in EDV.
  `ssm` (1.04 pp) and `prior_inr` (1.05 pp) give the best EF.
- **A third view helps most when there is no prior.** `rbf` EF MAE at the
  severe level falls from 2.50 pp (biplane) to 1.54 pp.
- **Direct `inr`.** It fails on the triplane layout, overestimating EDV by
  about 53 mL with perfect views (EF MAE 7.0 pp). The three planes meet along
  the long axis, and an unregularised field fills the space between them.

### Cavity Dice

Values are the mean of ED and ES. The unobserved-voxel Dice is within 0.02 of
the whole-volume Dice throughout; see `analysis/results.md`.

| Condition | `inr` | `rbf` | `ssm` | `unet` | `prior_inr` | `mean_shape` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| biplane, none | 0.870 | 0.953 | 0.930 | 0.961 | 0.954 | 0.659 |
| biplane, 5°/3 mm, known | 0.866 | 0.941 | 0.928 | 0.956 | 0.951 | 0.659 |
| biplane, 10°/5 mm, known | 0.810 | 0.909 | 0.923 | 0.932 | 0.943 | 0.659 |
| biplane, 5°/3 mm, assumed | 0.864 | 0.938 | 0.918 | 0.939 | 0.935 | 0.659 |
| biplane, 10°/5 mm, assumed | 0.834 | 0.888 | 0.869 | 0.883 | 0.880 | 0.659 |
| triplane, none | 0.764 | 0.973 | 0.930 | 0.973 | 0.963 | 0.659 |
| triplane, 10°/5 mm, known | 0.776 | 0.951 | 0.925 | 0.954 | 0.953 | 0.659 |

- **Fitting.** Every fitted method beats the unfitted mean cavity (0.659) by
  a wide margin.
- **Dice and EF disagree.** `ssm` has lower Dice than `unet` and `prior_inr`
  (0.92–0.93, because 20 modes limit detail), yet it gives the most accurate
  EF under misplacement. A plausible reason is that its errors are similar
  at ED and ES, so they largely cancel in the volume ratio; this was not
  tested. Overlap and the clinical endpoint do not
  rank the methods the same way.

## Compute

- **Model building.** The 4 cavity SSMs and 4 U-Nets took 1 h 53 min.
- **Evaluation.** The 4 evaluation units took 7 h 1 min. Median seconds per
  reconstruction: `simpson` 0.2, `mean_shape` 0.02, `unet` 0.08, `ssm` 1.1,
  `inr` 4.6, `rbf` 5.1, `prior_inr` 9.1.

## Notes

- **Simpson figures in the protocol's question.** The protocol quotes Simpson
  EF errors of 0.7, 2.0, and 4.9 pp from the earlier development check. That
  check averaged each subject's EF over two scans, which are two independent
  misplacement draws, and so understates the error of a single measurement.
  Per scan on the same folds 1–4, it gives 0.68, 2.35, and 6.83 pp. These are
  consistent with the single-draw values here: 0.68, 2.68, and 5.98 pp. No
  setting changed. The clarification is in the protocol's deviation log.
- **Fixed-sign shift.** Following the frozen protocol, `perturb_plane` shifts
  each view a fixed distance along its own normal, always in the positive
  direction. Because the normal direction of each apical plane is arbitrary
  relative to anatomy, this is not a systematic anatomical bias.
- **Limitations.**
  - *Inputs.* The views are label sections from 3D labels, not segmentations
    of 2D echo images.
  - *View placement.* The long axis comes from each phase's own label (oracle
    placement).
  - *Settings.* Method settings were transferred from the v2 union target
    without retuning.
  - *Interpretation.* The results show what each estimator can do given
    accurate tracings, not clinical accuracy.
