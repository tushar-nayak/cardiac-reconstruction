# Plan: full-cohort sparse benchmark → arXiv v1

Goal: turn the manuscript in `../heart-maker-report` from a one-subject pilot
into a defensible preprint by running a subject-level sparse-plane benchmark on
MITEA.

Planning facts from existing runs:

- **Runtime:** about 10 s per fit plus evaluation. Three methods × 108 subjects
  × 3 seeds ≈ 3–4 h on the Quadro RTX 4000.
- **Disk:** about 17 MB per saved occupancy grid, so roughly 17 GB if every run
  saved one. Save metrics only by default, and keep grids for a few figure
  subjects.

## Phase 0: Fix the protocol before running anything (½ day)

Write `benchmarks/protocol_v1.md` and do not change it afterwards. It should fix:

- **Subjects:** use the frozen split `benchmarks/splits/mitea_cohort_v1.json`
  (26 test subjects, 5 development folds, stratified by pathology).
  - Tune only on the fold-0 validation subjects (22).
  - Report on the other 86 development subjects. Run the 26 test subjects once,
    at the end, if the test-set decision below allows it.
- **Volumes:** scan 1, ED, union label (cavity + myocardium). This matches the
  current draft; cavity-only and ES are for the follow-up paper.
- **Plane layouts:** 3 central orthogonal planes is primary; 3 axial planes is
  secondary.
- **Primary metric:** unobserved-voxel Dice.
- **Secondary metrics:** whole Dice, HD95, ASSD, volume error, component count,
  ΔK.
- **Fixed settings:** threshold 0.5, 1000 steps, the pilot's learning rate and
  initialization. Hyperparameters may only be changed using fold 0.
- **Statistics:** average seeds within each subject, then run paired Wilcoxon
  signed-rank tests (Holm-corrected) and report 95% bootstrap CIs on the paired
  differences.

## Phase 1: Code changes (1–2 days, uncommitted)

1. **`scripts/benchmark_sparse.py`:** add `--manifest` and
   `--subset {fold0,development,test}` options that select subjects from the
   cohort manifest instead of the old 80/10/10 loader split.
2. **Surface metrics:** add HD95, ASSD and Hausdorff to
   `cardiac_common/sparse.py::reconstruction_metrics`, using
   `cardiac_common/clinical.py::surface_distances`.
3. **Interpolation baselines**, as a third `--methods` option:
   - *Orthogonal layout:* a visual hull, meaning a voxel is foreground only if
     its projections onto all three observed planes are foreground.
   - *Axial layout:* shape-based interpolation, which linearly blends the
     signed distance fields of neighbouring slices.
4. **Largest-component post-processing** as an extra reported variant, never
   the primary result. It tests whether INR fragments are tiny debris or a real
   failure.
5. **`scripts/summarize_benchmark.py`:** compute per-subject means, paired tests
   and bootstrap CIs, and write LaTeX-ready tables.
6. **Tests:**
   - The visual hull of a sphere's three central planes equals the expected
     solid.
   - Shape-based interpolation between identical slices gives a straight
     extrusion.
   - The subset loader never returns a test subject during development.

## Phase 2: Pilot on fold 0 (½ day of compute)

- Run all 3 methods × 22 subjects × 3 seeds, with timing.
- Sanity checks: no NaNs, no empty predictions, and the visual hull should come
  out conservative (under-estimates volume).
- If one method is clearly broken, fix it **here only**, then freeze the
  settings.

## Phase 3: Full runs (one evening of compute)

- **Development:** the remaining 86 subjects, orthogonal layout first, then
  axial.
- **Test:** the 26 subjects, once, with the identical command (only if the
  test-set decision allows it).
- Save occupancy grids only for 3 figure subjects. MITEA_071, MITEA_018 and
  MITEA_052 keep the draft's existing figures valid.

## Phase 4: Analysis (1–2 days)

- **Main table:** method × layout, median and IQR, with paired p-values against
  INR.
- **Subgroups by pathology:** healthy vs hypertrophy/HCM vs amyloid vs DCM.
  Keep this descriptive, since groups are small.
- **Figures:**
  - box/strip plots of per-subject Dice
  - Bland–Altman for volume error
  - component-count histogram, raw vs largest component
  - best/median/worst subject gallery
- Write down the one-sentence finding before writing any prose.

## Phase 5: Rewrite the manuscript (3–5 days)

- New framing: **"A subject-level benchmark for sparse-plane cardiac shape
  completion on MITEA with topology-aware evaluation."**
- Results section 1 becomes the n=108 benchmark. Mixed/Meta moves to an
  appendix as historical, non-reproducible results, or is cut.
- Cut the defensive hedging by roughly half; one Limitations section carries it.
- Regenerate figures with `make_figures.py` pointed at the new runs.

## Phase 6: Submission gate (runs in parallel; start now)

- [ ] MITEA permission for derived figures (email the providers first)
- [ ] All three authors approve; corresponding email, funding and conflicts
      filled in
- [ ] Bibliography checked against primary sources, especially the 2026 entries
- [ ] Code release with the protocol document and split manifest, as a tagged
      commit
- [ ] Source bundle: `main.tex`, `.bbl`, figures; exclude
      `L3DReport-Final.pdf` and any raw data
- [ ] Category: eess.IV primary, cs.CV cross-list

**Total: about 2–3 weeks.** The critical path is Phases 1–3 (about a week). The
MITEA permission reply may become the actual bottleneck.

## Open decisions

1. **Test-set policy:** reporting the held-out test set in the arXiv version
   spends it for the later MICCAI paper. Recommendation: keep the test set
   untouched and report 5-fold development results (n=108).
2. **Gaussian method:** keep it as a secondary method. It costs nothing extra to
   run and sets up the follow-up paper.

## Already in place (uncommitted)

- `cardiac_common/cohort.py`: volume enumeration, multi-class labels
  (1 = myocardium, 2 = LV cavity), stratified subject-level splits.
- `cardiac_common/views.py`: LV long axis, simulated apical planes, pose
  perturbation, plane sampling.
- `cardiac_common/clinical.py`: HD95/ASSD, volumes, EF, biplane Simpson.
- `scripts/build_cohort.py`, `scripts/simpson_baseline.py`.
- `benchmarks/splits/mitea_cohort_v1.json`: frozen split.
- `tests/test_cohort_views.py`: 8 tests, all passing.

Biplane Simpson baseline on the 108 development subjects (two apical planes 90°
apart, the two scans averaged per subject). Results are in `runs/simpson_*_dev_v01/`.

| View placement | EF MAE | EF limits of agreement | EDV bias |
| --- | --- | --- | --- |
| Perfect | 0.7% | −0.6 to +1.8% | −3.7 mL |
| 5° tilt, 3 mm shift | 2.0% | −2.1 to +5.3% | −9.7 mL |
| 10° tilt, 5 mm shift | 4.9% | −7.4 to +13.9% | −23.9 mL |

With perfectly placed views Simpson's method is nearly unbeatable on MITEA, so
the follow-up reconstruction paper has to win under realistic view misplacement.
