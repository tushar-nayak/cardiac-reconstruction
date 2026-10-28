# MITEA NeRF synthetic-rendering ablation v1

## Status and interpretation

This frozen, development-only ablation studies a NeRF-style density/radiance
field for **synthetic orthographic volume rendering**. MITEA supplies
reconstructed 3D intensity volumes and aligned masks, not posed 2D camera
images. For each subject, the full 3D scan and mask are volume-rendered to
create targets. The experiment therefore does not measure sparse-plane
reconstruction, does not use a NeRF trained from observed MITEA slices, and
cannot support clinical or raw-echocardiography claims. It is a method and
renderer feasibility study kept separate from the primary INR/Gaussian
manuscript.

No subject in the held-out test split is used. All runs are per-subject and
use only development fold 0. Eight subjects are used for screening and ten
disjoint fold-0 subjects for a limited confirmation of the selected settings.
The remaining fold-0 subjects are not used. This is still exploratory and is
not a final independent-cohort confirmation.

## Synthetic view generation and renderer

For each view, rays are orthographic and traverse the full normalized volume
box. The target density is fixed to $18Y(\mathbf{x})$, where $Y$ is the union
of the MITEA LV cavity and myocardium labels; grayscale radiance is the
min-max-normalized MITEA ultrasound intensity. Targets use emission-absorption
compositing with 64 uniformly spaced samples unless the sample-count factor is
being ablated. The model predicts softplus density and sigmoid grayscale
radiance from Fourier-encoded location and (in relevant variants) view
direction. The objective is rendered-intensity MSE plus a weighted rendered
opacity MSE. Both target types are synthetic projections derived from the
complete 3D scan.

The common novel-view endpoint renders the three reverse cardinal directions
$(-x,-y,-z)$, none of which is used for training. For each direction, report
grayscale PSNR and Dice of rendered opacity thresholded at 0.5 against the
target opacity thresholded at 0.5. Subject is the unit of summary; average
views and seeds within subject before cohort summaries. RGB PSNR is secondary;
opacity Dice is the screening selection metric. This metric evaluates
synthetic rendered silhouettes, not voxelwise 3D occupancy.

## Subjects and run schedule

Subject lists follow the checked-in fold-0 manifest order:

- Screening (first eight): `MITEA_007`, `MITEA_008`, `MITEA_009`,
  `MITEA_011`, `MITEA_016`, `MITEA_030`, `MITEA_042`, `MITEA_045`.
- Confirmation (next ten): `MITEA_048`, `MITEA_051`, `MITEA_064`,
  `MITEA_074`, `MITEA_083`, `MITEA_091`, `MITEA_094`, `MITEA_096`,
  `MITEA_116`, `MITEA_123`.
- Seeds: 17, 23, and 42.

Screen all configurations on the eight screening subjects. Select the two
highest screening mean opacity-Dice configurations; ties are resolved by
higher mean PSNR, then lower parameter count. Confirm those two and the frozen
baseline on the ten confirmation subjects, with all three seeds. The
confirmation stage does not retune any setting. The baseline and all listed
variants are fixed below before the screening run.

## Ablation factors

Baseline: 3 positive cardinal training views ($+x,+y,+z$), position encoding
frequency 8, view-direction encoding frequency 4, width 128, depth 4, 64
samples per ray, opacity-loss weight 0.25, Adam learning rate $5\times10^{-4}$,
1,200 steps, and 256 rays per update. Each variant changes one factor:

| Factor | Fixed variants |
| --- | --- |
| View count | 1 view ($+z$); 2 views ($+x,+z$); 3 views (baseline) |
| Position frequencies | 4; 12 |
| View direction | disabled; frequency 2 (baseline frequency 4) |
| Capacity | width 64/depth 4; width 192/depth 6 |
| Samples per ray | 32; 128 |
| Opacity loss weight | 0; 1.0 |
| Learning rate | $2.5\times10^{-4}$; $10^{-3}$ |
| Optimization steps | 600; 2,400 |

All other settings remain at baseline. The ablation is one-factor-at-a-time;
it estimates local sensitivity around the selected baseline and does not
identify interactions among factors. View-count subsets are nested, and all
three training directions are cardinal; angular clustering is not studied.

## Reproducibility

Run `scripts/run_nerf_ablation.py`. The script records per-run settings,
training loss, held-out render metrics, elapsed time, and confirmation
checkpoints beneath the git-ignored `runs/nerf_mitea_ablation_v1/` directory.
Use `--resume` to continue an interrupted frozen run. Do not edit the protocol
or model settings during the run. The earlier single-subject renderer smoke
test is described in `nerf_mitea_pilot.md` and is not pooled with this study.
