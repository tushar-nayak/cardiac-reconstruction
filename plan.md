# Plan: full-cohort sparse benchmark → arXiv v1

Goal: turn the manuscript in `../heart-maker-report` from a one-subject pilot
into a defensible preprint by running a subject-level sparse-plane benchmark on
MITEA.

Planning facts from existing runs:

- **Runtime (timed on MITEA_007, Quadro RTX 4000):** per seed, INR takes about
  9 s (fit, grid evaluation and metrics), Gaussian about 13 s, and interpolation
  about 2 s once per subject. That is roughly 70 s per subject and layout for
  INR + Gaussian + interpolation with 3 seeds, and 40 s for an extra Gaussian
  recipe.
- **Disk:** about 17 MB per saved occupancy grid, but only about 150 KB per
  subject for metrics and checkpoints. Save grids only for the figure subjects.

## Phase 0: Fix the protocol before running anything (½ day)

Write `benchmarks/protocol_v1.md` and do not change it afterwards. **Complete:**
the frozen protocol is in place. It resolves the decisions below and adds a
physical-spacing provenance check. It fixes:

- **Subjects:** use the frozen split `benchmarks/splits/mitea_cohort_v1.json`
  (26 test subjects, 5 development folds, stratified by pathology).
  - Tune only on the fold-0 validation subjects (22).
  - Use fold 0 (22 subjects) only for development/selection and report on the
    other 86 development subjects. Keep the 26 test subjects untouched.
- **Volumes:** scan 1, ED, union label (cavity + myocardium). This matches the
  current draft; cavity-only and ES are for the follow-up paper.
- **Plane layouts:** 3 central orthogonal planes is primary; 3 axial planes is
  secondary, at depth fractions 0.375/0.5/0.625 per Amendment 1 in the
  protocol's deviation log.
- **Primary metric:** unobserved-voxel Dice.
- **Secondary metrics:** whole Dice, HD95, ASSD, volume error, component count,
  ΔK.
- **Fixed settings:** threshold 0.5, 1000 steps, and three fixed seeds; the only
  recipe selection is the predeclared Gaussian choice on fold 0.
- **Statistics:** average seeds within each subject, then run paired Wilcoxon
  signed-rank tests (Holm-corrected) and report 95% bootstrap CIs on the paired
  differences.

## Phase 1: Code changes (1–2 days, uncommitted)

**Complete.** 52 tests pass. The axial-layout problem found here is logged as
Amendment 1 in `benchmarks/protocol_v1.md`.

1. **`scripts/benchmark_sparse.py`:** `--manifest` with
   `--subset {fold0,…,fold4,confirmatory}` selects subjects from the cohort
   manifest; test subjects cannot be selected. Also `--subjects` (smoke
   checks), `--save-occupancy-subjects`, `--axial-fractions`, and failure
   recording. Without `--manifest` the legacy behaviour is unchanged.
2. **Surface metrics:** add HD95, ASSD and Hausdorff to
   `cardiac_common/sparse.py::reconstruction_metrics`, using
   `cardiac_common/clinical.py::surface_distances`.
3. **Interpolation baselines**, as a third `--methods` option:
   - *Orthogonal layout:* section-extrusion intersection (SEI), meaning a voxel
     is foreground only if its in-plane coordinates are foreground in all
     three observed cross-sections. This is not the silhouette-based visual
     hull and has no guaranteed volume-bias direction for section data.
   - *Axial layout:* shape-based interpolation, which linearly blends the
     signed distance fields of neighbouring slices.
4. **Largest-component post-processing** as an extra reported variant, never
   the primary result. It tests whether INR fragments are tiny debris or a real
   failure.
5. **`scripts/summarize_benchmark.py`:** compute per-subject means, paired tests
   and bootstrap CIs, and write LaTeX-ready tables.
6. **Tests:**
   - The SEI of a sphere's three central planes equals the analytically
     computed intersection of three extruded disks (not the sphere itself).
   - Shape-based interpolation between identical slices gives a straight
     extrusion.
   - The subset loader never returns a test subject during development.

## Phase 2: Pilot on fold 0 (about 1.5 h of compute)

Fold 0 is the only place where settings may change. Run everything from the
repository root. Every `--run-dir` must be new; the runner refuses to
overwrite.

**Step 1: run both recipes on both layouts.** Run A uses the default
uniform Gaussian recipe (learning rate 0.001, shared with INR). Run B is the
observed-initialization recipe, in its own run because its learning rate differs
from INR's.

```bash
M=benchmarks/splits/mitea_cohort_v1.json
for L in orthogonal axial; do
  F=$([ $L = axial ] && echo "--axial-fractions 0.375 0.5 0.625")
  .venv/bin/python scripts/benchmark_sparse.py --data-dir dataset/mitea \
    --manifest $M --subset fold0 --plane-layout $L $F \
    --run-dir runs/protocol_v1_fold0_${L}_A \
    --methods inr gaussian interpolation --seeds 17 23 42 --device cuda
  .venv/bin/python scripts/benchmark_sparse.py --data-dir dataset/mitea \
    --manifest $M --subset fold0 --plane-layout $L $F \
    --run-dir runs/protocol_v1_fold0_${L}_B \
    --methods gaussian --gaussian-init observed --lr 0.003 \
    --seeds 17 23 42 --device cuda
done
```

**Step 2: sanity checks.** Summarize fold 0 (allowed: it is the development
fold) and check failures and empty predictions:

```bash
.venv/bin/python scripts/summarize_benchmark.py report \
  --runs runs/protocol_v1_fold0_orthogonal_A runs/protocol_v1_fold0_axial_A \
  --expect-subset fold0 --output runs/protocol_v1_fold0_sanity.json
.venv/bin/python -c "
import json, glob
for path in sorted(glob.glob('runs/protocol_v1_fold0_*/results.json')):
    runs = json.load(open(path))['runs']
    failed = [r for r in runs if r['status'] == 'failed']
    empty = [(r['method'], r['subject'], r['seed']) for r in runs
             if r['status'] == 'ok' and r['empty_prediction']]
    print(path, len(runs), 'runs;', len(failed), 'failed;', 'empty:', empty)"
```

Expect 22 subjects per layout, 3 seeds for learned methods, one deterministic
run for interpolation, and no failures. Empty predictions or a summary
`missing` count above zero need an explanation before moving on. SEI
correctness is already covered by `tests/test_interpolation.py`; do not assume
it under- or over-estimates volume.

**Step 3: select the Gaussian recipe** with the predeclared rule:

```bash
.venv/bin/python scripts/summarize_benchmark.py select-gaussian \
  --uniform runs/protocol_v1_fold0_orthogonal_A runs/protocol_v1_fold0_axial_A \
  --observed runs/protocol_v1_fold0_orthogonal_B runs/protocol_v1_fold0_axial_B \
  --output runs/protocol_v1_fold0_gaussian_selection.json
```

**Step 4: freeze.** Copy the `selected` value and both recipe scores into the
protocol's deviation log as a dated "Fold-0 selection" entry, before running
any fold-1–4 command. If a method is broken, fix it now, log the fix, and rerun
**all** of fold 0 (Steps 1–3) into new run directories.

## Phase 3: Confirmatory runs (about 3.5 h of compute)

Runs the 86 subjects in folds 1–4 once, with settings frozen in Phase 2. Do not
inspect partial results to decide whether to continue, and never pass
`--subset test`.

**Step 1: INR and interpolation**, both layouts, with dense grids saved only
for the three figure subjects (all in folds 1–4):

```bash
M=benchmarks/splits/mitea_cohort_v1.json
FIG="MITEA_071 MITEA_018 MITEA_052"
for L in orthogonal axial; do
  F=$([ $L = axial ] && echo "--axial-fractions 0.375 0.5 0.625")
  .venv/bin/python scripts/benchmark_sparse.py --data-dir dataset/mitea \
    --manifest $M --subset confirmatory --plane-layout $L $F \
    --run-dir runs/protocol_v1_confirmatory_${L}_inr_interp \
    --methods inr interpolation --seeds 17 23 42 \
    --save-occupancy-subjects $FIG --device cuda
done
```

**Step 2: Gaussian with the selected recipe**, in its own run directories.
Set `G` from `runs/protocol_v1_fold0_gaussian_selection.json`:

```bash
# Selected "uniform_lr0.001":
G="--gaussian-init uniform --lr 0.001"
# Selected "observed_lr0.003":
# G="--gaussian-init observed --lr 0.003"
for L in orthogonal axial; do
  F=$([ $L = axial ] && echo "--axial-fractions 0.375 0.5 0.625")
  .venv/bin/python scripts/benchmark_sparse.py --data-dir dataset/mitea \
    --manifest $M --subset confirmatory --plane-layout $L $F \
    --run-dir runs/protocol_v1_confirmatory_${L}_gaussian \
    --methods gaussian $G --seeds 17 23 42 \
    --save-occupancy-subjects $FIG --device cuda
done
```

**Step 3: confirmatory report.** It checks that every method saw identical
observations per subject and layout, averages seeds within subjects, and writes
JSON plus a LaTeX table:

```bash
.venv/bin/python scripts/summarize_benchmark.py report \
  --runs runs/protocol_v1_confirmatory_orthogonal_inr_interp \
         runs/protocol_v1_confirmatory_axial_inr_interp \
         runs/protocol_v1_confirmatory_orthogonal_gaussian \
         runs/protocol_v1_confirmatory_axial_gaussian \
  --expect-subset confirmatory \
  --output runs/protocol_v1_confirmatory_report.json
```

The four Holm-corrected comparisons (Gaussian and interpolation vs INR, per
layout) are printed and saved under `paired_vs_inr`. Failed runs are listed
under `failed_runs` and must be reported with their denominators, not dropped.

**If a run is interrupted:** do not resume into the same directory. Rerun the
affected layout into a new directory and report the replacement in the deviation
log.

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
- Results section 1 becomes the n=86 confirmatory benchmark. Fold 0 is used for
  selection and is not included in confirmatory statistics. Mixed/Meta moves to an
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

1. **Test-set policy (resolved):** keep the held-out test set untouched for the
   later study; report the 86 development subjects not used for fold-0 tuning.
2. **Gaussian method (resolved):** keep it secondary and compare only the
   predeclared recipes/settings in the protocol.

## Already in place (uncommitted)

- `cardiac_common/cohort.py`: volume enumeration, multi-class labels
  (1 = myocardium, 2 = LV cavity), stratified subject-level splits.
- `cardiac_common/views.py`: LV long axis, simulated apical planes, pose
  perturbation, plane sampling.
- `cardiac_common/clinical.py`: HD95/ASSD, volumes, EF, biplane Simpson.
- `scripts/build_cohort.py`, `scripts/simpson_baseline.py`.
- `benchmarks/splits/mitea_cohort_v1.json`: frozen split.
- `cardiac_common/interpolation.py`: SEI and axial signed-distance
  interpolation baselines, reading observed sections only.
- `cardiac_common/sparse.py`: surface distances, component counts,
  largest-component sensitivity, `select_planes(..., fractions)`.
- `scripts/benchmark_sparse.py`: manifest-based runner (see Phase 1).
- `scripts/summarize_benchmark.py`: `select-gaussian` and `report`.
- `benchmarks/protocol_v1.md`: frozen protocol with deviation log.
- Tests: `tests/test_cohort_views.py`, `tests/test_interpolation.py`; 52 tests
  in the suite, all passing.

Biplane Simpson baseline on the 108 development subjects (two apical planes 90°
apart, the two scans averaged per subject). Results are in `runs/simpson_*_dev_v01/`.

| View placement | EF MAE | EF limits of agreement | EDV bias |
| --- | --- | --- | --- |
| Perfect | 0.7% | −0.6 to +1.8% | −3.7 mL |
| 5° tilt, 3 mm shift | 2.0% | −2.1 to +5.3% | −9.7 mL |
| 10° tilt, 5 mm shift | 4.9% | −7.4 to +13.9% | −23.9 mL |

With perfectly placed views Simpson's method is nearly unbeatable on MITEA, so
the follow-up reconstruction paper has to win under realistic view misplacement.
