# FeEcho4D External Radial-View Results

**Evaluation date:** 2026-10-23
**Protocol:** [`feecho4d_external_protocol.md`](feecho4d_external_protocol.md)
**Run artifacts:** `runs/feecho4d_external_20261023_final/` (local, Git-ignored)

## Cohort and task

The annotated cohort contains 52 fetal subjects: 47 healthy and 5 with
congenital heart disease (2 ToF, 3 dTGA). At each subject's metadata-defined
ED frame, masks from radial views 7, 19, and 31 (30°, 90°, and 150°) were used
for fitting; the other 34 views were held out. Foreground is the union of LV
cavity and myocardium. Input is binary mask geometry only; ultrasound
intensities are not used.

The distributed archive omits per-subject rotation-center metadata. Therefore,
coordinates are normalized pixel x/y and view angle, rather than physical
Cartesian coordinates. These results measure withheld radial-mask completion,
not 3D surface reconstruction or clinical validity. They are not pooled with
MITEA's native-grid unobserved-voxel results.

## Results

All 312 planned runs completed (52 subjects × 2 methods × 3 seeds), with no
failures. The table reports seed-averaged subject scores, then cohort
summaries.

| Method | Pooled withheld-view Dice, mean ± SD | Median [IQR] | Successful runs |
| --- | ---: | ---: | ---: |
| INR | 0.918 ± 0.015 | 0.920 [0.908, 0.930] | 156/156 |
| Gaussian occupancy | 0.805 ± 0.029 | 0.802 [0.779, 0.828] | 156/156 |

Paired Gaussian-minus-INR mean difference was −0.1126 Dice (95% subject
bootstrap CI −0.1198 to −0.1048; two-sided Wilcoxon signed-rank p =
3.50 × 10⁻¹⁰). This single comparison is exploratory. The descriptive
diagnostic subgroup is small (n = 5): mean pooled Dice was 0.912 for INR and
0.792 for Gaussian occupancy; no subgroup inference is made.

## Reproduction

The extraction directory and run records are local and Git-ignored. With the
archive extracted under `dataset/FeEcho4D_extracted/`, rerun using:

```bash
./.venv/bin/python scripts/evaluate_feecho4d.py \
  --archive dataset/FeEcho4D_extracted \
  --output-dir runs/feecho4d_external_REPRODUCED
./.venv/bin/python scripts/summarize_feecho4d.py \
  runs/feecho4d_external_REPRODUCED
```

The source data are not redistributed. FeEcho4D is used under its maintainers'
noncommercial research terms; cite Hasan et al., *IEEE Transactions on Medical
Imaging*, 2026, doi:10.1109/TMI.2026.3707322, and the [dataset project
page](https://feecho4d.github.io/Website/).
