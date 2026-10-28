# FeEcho4D interpolation, annotation count, and angular coverage

**Frozen on 2026-10-23 before running these baselines or ablations.** This is
an exploratory extension after the original 52-subject INR/Gaussian results
were inspected (mean pooled Dice 0.918 and 0.805). It is not a new untouched
cohort or independent confirmation. No outcome-dependent tuning is permitted.

## Question and controls

Does the current INR add accuracy over simple interpolation, and how do the
number and distribution of annotations affect completion? A baseline win or
tie will be reported with the same prominence as an INR win. No methods,
subjects, layouts, metrics, or thresholds will be removed based on scores.

Use the same 52 annotated subjects, metadata-defined ED frame, native mask
pixels, foreground union (127 or 255), and normalized (x,y,angle) coordinates
as the original protocol. Source images, meshes, unannotated subjects, and
population training remain unused. This is radial annotation completion,
not physical Cartesian reconstruction or clinical validation.

## Fixed observations and common evaluation views

Angles are degrees; file indices are angle / 5 + 1.

| Layout | Observed angles | Count | Angular span |
| --- | --- | ---: | ---: |
| spread2 | 30, 150 | 2 | 120 |
| spread3 | 30, 90, 150 | 3 | 120 |
| spread6 | 0, 30, 60, 90, 120, 150 | 6 | 150 |
| spread12 | 0, 15, 30, ..., 165 | 12 | 165 |
| clustered3 | 75, 90, 105 | 3 | 30 |

The spread layouts have nested observations: each larger budget contains the
smaller budget. Counts 6 and 12 also expand the observed span; count effects
must not be attributed solely to count. The spread3/clustered3 comparison
holds annotation count fixed while changing coverage.

**Primary evaluation:** the same 24 views for every layout: all source angles
that are not multiples of 15 degrees (5, 10, 20, 25, ..., 170, 175). These never
overlap any observation, and exclude the geometrically duplicated 0/180 pair.
The primary score pools foreground pixel counts across these views per
subject. Do not compare different view counts using only their changing
complements of observed views.

**Secondary:** mean per-view Dice and IoU on the common views; common-view
Dice restricted to 30--150 degrees (16 identical views); common views inside
versus outside the observed angular range; and Dice stratified by distance
to the nearest observation (0--15, >15--30, >30--60, >60 degrees). These are
descriptive subject-level scores; views/pixels are not statistical replicates.

Also save scores on every unobserved source view. For spread3 this is the
original 34-view endpoint and serves as a reproduction check against the
original saved results. It is secondary to the common 24-view endpoint here.

## Fixed methods

1. **Nearest:** copy the mask of the closest observed angle; ties choose the
   lower angle. Beyond the observed angular range, copy the closest endpoint.
2. **SDF interpolation (primary comparator):** compute positive-inside 2D
   signed distance in pixel units for each observed mask, then linearly
   interpolate at the query angle and threshold at zero. Outside the observed
   range, extend the nearest observed field. Empty/full masks use the existing
   constant-distance convention. No alignment, smoothing, or periodic wrapping.
3. **INR:** original 64-width, four-layer, four-frequency network.
4. **Gaussian occupancy:** original 256 primitives and uniform initialization.

Both learned fields retain 1,000 Adam steps, batch 4,096 uniformly sampled
observed pixels, learning rate 0.001, threshold 0.5, and seeds 17, 23, 42.
Training starts from scratch per subject/layout/seed. No early stopping or
new initialization recipes. Both deterministic baselines run once per layout
and subject. Every paired method must have identical observed-mask and
evaluation-grid fingerprints. Baseline prediction functions receive only
observed masks and their angles. Targets are used only for scoring.

## Analysis and artifacts

- Plan: 52 subjects x 5 layouts x (2 baseline runs + 6 learned runs) = 2,080
  records, including 1,560 learned fits. Retain failed/empty predictions and
  report coverage; do not replace missing seeds by zeros or partial averages.
- Average all three learned seeds within subject before any comparisons.
- Report mean/sample SD, median/IQR, subject scores, fit and query timings.
- Primary comparator is fixed as SDF, not whichever baseline wins. For each
  of five layouts, report paired SDF-minus-INR differences on common-view
  pooled Dice, 95% subject-bootstrap CIs (10,000 replicates, seed 2027), and
  two-sided paired Wilcoxon tests with Holm correction across these five.
  All inference remains exploratory.
- Nearest and Gaussian comparisons, incremental annotation-count effects,
  and spread3-minus-clustered3 effects are secondary/descriptive with paired
  subject-bootstrap intervals. No post-hoc superiority or equivalence margin.
- Retain the original evaluation outputs unchanged. Save new records under
  a separate ignored run directory; archive protocol/source hashes and
  software versions. Concurrent GPU workers are allowed for throughput;
  wall-clock timings under contention are descriptive, not speed benchmarks.
- Qualitative examples, if saved, are preselected as Patient001/Patient016,
  seed 17, query angles 5, 65, 115, 175 degrees. These do not select methods.
- Generate complete tables and aggregate plots from saved records, and update
  manuscript claims to match the observed ranking, including null/negative
  results. Do not pool this endpoint with MITEA or the original 34-view result.
