# MITEA Apical-View Ejection Fraction Study v1

**Status: frozen (2026-11-02) before any run on the evaluation cohort.**
Smoke checks used only the fold-0 subject MITEA_007 (see the deviation log).
No setting may change after evaluation-cohort outputs are inspected.

## Question

Clinical LV volumes and ejection fraction (EF) are measured from two apical
views by the biplane method of disks (Simpson). With perfectly placed views
Simpson is nearly exact on MITEA labels. In the 108-subject development check
(`runs/simpson_*_dev_v01/`), its EF mean absolute error (MAE) was 0.7
percentage points with perfect views, 2.0 at 5° tilt / 3 mm shift, and 4.9 at
10° / 5 mm.

The question is whether 3D reconstruction from the **same** apical views gives
more accurate cavity volumes and EF, especially when the views are misplaced.
Two cases are considered: the true plane pose is known (for example from probe
tracking or a 3D acquisition), or the intended pose is assumed.

This is a development-cohort study. It was designed after the v2 test and the
sensitivity analyses were known, and it reports estimates with confidence
intervals but no hypothesis tests. It is not a clinical validation: inputs are
label sections extracted from 3D labels, not segmented 2D echo images.

## Data and cohort

- **Manifest and cohort.** Frozen manifest
  `benchmarks/splits/mitea_cohort_v1.json`. Evaluate development folds 1–4
  (86 subjects). The test set is not loaded.
- **Volumes.** Scan 1, end-diastole (ED) and end-systole (ES). Both scans of a
  subject share near-identical CMR-derived labels, so scan 2 adds no
  independent information.
- **Target.** The LV cavity (label 2). True EDV, ESV, and EF come from the
  full 3D cavity labels.
- **Cross-fitted priors.** For evaluation fold k, population models are
  trained on the other four development folds using the scan-1 ED and ES
  cavity labels (two shapes per subject). The runner rejects any evaluated
  subject that appears in a model's training list.

## Views

- **Long axis and planes.** For each phase, the cavity long axis is estimated
  from that phase's own label (`lv_long_axis`, an oracle view placement, as in
  the Simpson baseline). Apical planes contain the long axis and are rotated
  about it (`apical_planes`).
- **Layouts.**
  - *Biplane (primary):* angles 0° and 90°.
  - *Triplane (secondary):* angles 0°, 60°, and 120°.
- **Misplacement.** Each plane is tilted by a fixed angle about a random
  in-plane axis and shifted along its normal (`perturb_plane`). There are
  three levels: none; 5° / 3 mm; 10° / 5 mm. Draws use
  `numpy.random.SeedSequence([2027, subject number, phase index, layout
  index, level index])`. They are shared by Simpson, by every method, and by
  both pose conditions.
- **Pose conditions.**
  - *Known pose:* each observed voxel lies within 0.5 mm of an actual
    (misplaced) plane and carries the 3D label at that voxel.
  - *Assumed pose:* observed voxels lie on the intended planes. Each carries
    the label read from the misplaced plane at the same in-plane position
    (nearest neighbour), which models a misplaced image placed as if it were
    the intended view. Where intended planes overlap, the first-listed plane's
    label is used.
- **Conditions run.**
  - Biplane: no misplacement; known pose at both levels; assumed pose at both
    levels (five conditions).
  - Triplane: no misplacement, and 10° / 5 mm with known pose.

## Methods

The frozen v2 settings carry over without retuning to the cavity target. This
transfer is a stated limitation. Stochastic methods use seed 17.

- **`simpson`:** biplane method of disks (20 disks, 0.5 mm in-plane sampling)
  on the two misplaced views. Biplane layout only.
- **`inr`:** v1 direct INR fitted to the observed voxels.
- **`rbf`:** φ = r³, λ = 0.01. Constraints are in-plane signed distances of the
  cavity on each view's 1 mm grid, placed at the believed plane positions.
  Centres are chosen as in v2.
- **`ssm`:** PCA of cavity signed-distance maps (ED and ES of each training
  subject). K = 20, λ = 0.001.
- **`unet`:** v2 architecture and recipe (lr 3e-4, 4,000 iterations, seed 17).
  Each training input is a random training volume and phase, a random layout
  (biplane or triplane), and a random misplacement with tilt uniform in
  [0°, 10°] and shift uniform in [0, 5] mm, using known pose. Evaluation uses
  the exact views above.
- **`prior_inr`:** the selected `ssm` fit, distilled and then fine-tuned as in
  v2 (β = 0, η = 1e-4).
- **`mean_shape`:** the cavity SSM mean (ED and ES pooled), reported for Dice
  only. It has no phase information, so it gives no EF.

Gaussian occupancy, `gp`, and the layout-specific interpolation baselines are
not run, because Simpson is the non-learned clinical comparator.

## Endpoints and analysis

- **Unit of inference.** The subject. EF = 100 · (EDV − ESV) / EDV, from the
  predicted cavity volumes (voxels with occupancy ≥ 0.5) at ED and ES under the
  same layout, condition, and method.
- **Primary.** Biplane layout with known pose, at each misplacement level: the
  paired difference in absolute EF error, method minus Simpson. Negative
  values favour the method. Report the mean with a 95% subject-bootstrap CI
  (10,000 replicates, seed 2027) and the number of subjects improved.
- **Secondary.**
  - EF and volume agreement (bias, MAE, 95% limits of agreement) for every
    method and condition.
  - The assumed-pose conditions.
  - The triplane layout.
  - Cavity Dice, whole-volume and unobserved-voxel.
- **Materiality.** A difference is material when its CI excludes 0 and its
  magnitude is at least 1 EF percentage point (5 mL for volumes).
- **No p-values.** Failures and empty predictions are reported. Within each
  subject, phase, layout, and condition, all voxel methods must share one
  observation fingerprint.

## Compute and reproducibility

- **Training:** 4 cavity SSMs and 4 U-Nets, about 2.5 h.
- **Evaluation:** about 14 reconstruction sets per subject, about 11 h.
- Steps run one at a time from `scripts/run_apical_ef_v1.sh`. Outputs go under
  the git-ignored `runs/mitea_apical_ef_v1/`, and finished units are skipped.
- Summary: `scripts/summarize_apical_ef.py`. Results:
  `benchmarks/mitea_apical_ef_v1_results.md`.

## Deviation log

### 2026-11-02 — Information available when drafting (disclosure)

The protocol was drafted after the v2 test, the sensitivity analyses, and the
development-set Simpson check were known. No reconstruction method has been
run on apical views, on the cavity target, or at ES.

### 2026-11-02 — Implementation and smoke checks before freezing

Implementation:

- `cardiac_common/apical.py`: views, the two pose conditions, RBF
  constraints, and the U-Net training observer;
- `--target`, `--phases`, and `--observation` options for
  `scripts/train_ssm.py` and `scripts/train_unet.py`;
- the runner `scripts/benchmark_apical.py`, the summary script
  `scripts/summarize_apical_ef.py`, and `tests/test_apical.py`.

Smoke checks used MITEA_007 with throwaway models trained on fold-1 subjects.
Outputs are in `runs/mitea_apical_ef_v1_smoke_20261102/`.

- With an 11-subject cavity SSM (21 modes) and a 30-iteration apical U-Net,
  all 94 rows (7 methods, 14 phase-by-condition sets) completed, each set with
  a single observation fingerprint.
- An earlier smoke SSM from 10 subjects had only 19 modes. That is fewer than
  K = 20, so the shape fit failed; the protocol models have about 170 modes.
- The smoke timings are not representative, because another project's jobs
  were using the GPU and CPU. The compute estimate above is unchanged.

No evaluation-cohort subject has been run.

### 2026-11-03 — Clarification after the run (no setting changed)

The Simpson EF errors quoted under *Question* (0.7, 2.0 and 4.9 percentage
points) average each subject's EF over two scans, which are two independent
misplacement draws. Per scan on folds 1–4 the same check gives 0.68, 2.35
and 6.83 points, consistent with this study's single-draw Simpson values.
The run completed with no deviation from the frozen settings.
