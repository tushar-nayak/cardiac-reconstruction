# MITEA NeRF ablation results

Exploratory study under the frozen
[ablation protocol](nerf_mitea_ablation_v1.md); outside the sparse-plane
manuscript. Run artifacts and figures are under the git-ignored
`runs/nerf_mitea_ablation_v1/analysis/`, produced by
`scripts/summarize_nerf_ablation.py` and `scripts/export_nerf_ablation.py`.
Development fold 0 only; no test subject was used.

**Scope:** synthetic orthographic rendering from complete 3D scans; this is not sparse-plane reconstruction.

Screening: 408 successful fits across 8 development subjects and 17 configurations.
Confirmation: 90 successful fits across 10 disjoint development subjects.
Selected after screening: viewdir_off, position_pe12.

Primary outcome is opacity Dice on the three held-out reverse cardinal views. Values average seeds within subject before cohort summaries.

## Screening

| Configuration | Mean Dice | SD | Median [IQR] | 95% bootstrap CI | Mean PSNR (dB) | Median fit (s) |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.9525 | 0.0102 | 0.9564 [0.9530, 0.9578] | [0.9452, 0.9575] | 23.71 | 26.6 |
| views1_z | 0.6726 | 0.0348 | 0.6730 [0.6549, 0.6889] | [0.6500, 0.6952] | 16.66 | 25.6 |
| views2_xz | 0.8510 | 0.0320 | 0.8491 [0.8263, 0.8758] | [0.8308, 0.8716] | 20.50 | 26.7 |
| position_pe4 | 0.9591 | 0.0042 | 0.9604 [0.9575, 0.9616] | [0.9561, 0.9615] | 23.59 | 26.9 |
| position_pe12 | 0.9720 | 0.0077 | 0.9736 [0.9666, 0.9766] | [0.9668, 0.9768] | 24.29 | 30.5 |
| viewdir_off | 0.9903 | 0.0018 | 0.9906 [0.9898, 0.9909] | [0.9891, 0.9914] | 26.98 | 25.4 |
| viewdir_pe2 | 0.9626 | 0.0091 | 0.9636 [0.9552, 0.9696] | [0.9568, 0.9685] | 24.05 | 27.3 |
| capacity_w64_d4 | 0.9249 | 0.0191 | 0.9248 [0.9165, 0.9400] | [0.9123, 0.9368] | 22.05 | 29.7 |
| capacity_w192_d6 | 0.9611 | 0.0138 | 0.9660 [0.9622, 0.9677] | [0.9511, 0.9678] | 23.77 | 28.6 |
| samples32 | 0.9542 | 0.0104 | 0.9584 [0.9526, 0.9597] | [0.9467, 0.9598] | 23.61 | 25.8 |
| samples128 | 0.9497 | 0.0105 | 0.9492 [0.9486, 0.9560] | [0.9423, 0.9557] | 23.40 | 28.4 |
| opacity_weight0 | 0.7789 | 0.0979 | 0.7883 [0.7144, 0.8398] | [0.7138, 0.8402] | 21.52 | 27.7 |
| opacity_weight1 | 0.9503 | 0.0120 | 0.9543 [0.9491, 0.9561] | [0.9416, 0.9565] | 23.30 | 27.4 |
| lr2p5e4 | 0.9472 | 0.0078 | 0.9485 [0.9418, 0.9528] | [0.9420, 0.9520] | 23.10 | 29.2 |
| lr1e3 | 0.9563 | 0.0087 | 0.9590 [0.9519, 0.9615] | [0.9503, 0.9614] | 23.78 | 27.0 |
| steps600 | 0.9465 | 0.0069 | 0.9477 [0.9440, 0.9518] | [0.9418, 0.9507] | 23.03 | 12.5 |
| steps2400 | 0.9515 | 0.0134 | 0.9545 [0.9464, 0.9598] | [0.9422, 0.9594] | 23.76 | 55.0 |

Screening paired differences versus baseline are exploratory because they are used for configuration selection.

## Disjoint development confirmation

| Configuration | Mean Dice | SD | Median [IQR] | 95% bootstrap CI | Mean PSNR (dB) |
|---|---:|---:|---:|---:|---:|
| baseline | 0.9438 | 0.0146 | 0.9440 [0.9376, 0.9507] | [0.9350, 0.9520] | 24.16 |
| position_pe12 | 0.9683 | 0.0085 | 0.9680 [0.9644, 0.9754] | [0.9634, 0.9734] | 25.13 |
| viewdir_off | 0.9900 | 0.0015 | 0.9897 [0.9893, 0.9903] | [0.9892, 0.9909] | 27.94 |

| Selected variant minus baseline | Mean paired Dice difference | 95% bootstrap CI | Holm p | Higher / n |
|---|---:|---|---:|---:|
| position_pe12 | +0.0246 | [+0.0199, +0.0299] | 0.00391 | 10/10 |
| viewdir_off | +0.0462 | [+0.0387, +0.0542] | 0.00391 | 10/10 |

These results concern synthetic view rendering from full-volume-derived targets. They do not estimate sparse-view 3D reconstruction, generalization to new subjects, or clinical performance.
