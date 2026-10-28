# FeEcho4D interpolation, annotation count, and coverage results

Exploratory extension; primary endpoint pools the same 24 withheld angular views in every condition. This analysis is archived separately and is outside the current MITEA-focused manuscript. It is retained here as a complete record rather than reporting only the earlier INR/Gaussian pilot.

Completed 2080/2080 planned runs; 0 failed, 0 empty predictions.

Run artifacts are locally available under the git-ignored
`runs/feecho4d_ablation_v1/analysis/`; the frozen protocol, runner, audit
script, and exporter are versioned alongside this report. FeEcho4D scores use
radial image/angle coordinates and are not physical 3D or clinical-validation
metrics.

| Layout | Method | Mean ± SD Dice | Median [IQR] | Subjects |
| --- | --- | --- | --- | ---: |
| spread2 | nearest | 0.9313 ± 0.0172 | 0.9349 [0.9200, 0.9444] | 52 |
| spread2 | sdf | 0.9358 ± 0.0187 | 0.9404 [0.9231, 0.9520] | 52 |
| spread2 | inr | 0.9040 ± 0.0216 | 0.9056 [0.8896, 0.9224] | 52 |
| spread2 | gaussian | 0.6124 ± 0.0378 | 0.6159 [0.5797, 0.6366] | 52 |
| spread3 | nearest | 0.9509 ± 0.0092 | 0.9520 [0.9448, 0.9573] | 52 |
| spread3 | sdf | 0.9560 ± 0.0078 | 0.9558 [0.9514, 0.9628] | 52 |
| spread3 | inr | 0.9240 ± 0.0128 | 0.9255 [0.9160, 0.9338] | 52 |
| spread3 | gaussian | 0.8296 ± 0.0278 | 0.8256 [0.8060, 0.8505] | 52 |
| spread6 | nearest | 0.9615 ± 0.0061 | 0.9621 [0.9575, 0.9663] | 52 |
| spread6 | sdf | 0.9661 ± 0.0054 | 0.9667 [0.9627, 0.9699] | 52 |
| spread6 | inr | 0.9411 ± 0.0089 | 0.9411 [0.9348, 0.9483] | 52 |
| spread6 | gaussian | 0.9070 ± 0.0149 | 0.9073 [0.8931, 0.9193] | 52 |
| spread12 | nearest | 0.9699 ± 0.0050 | 0.9705 [0.9671, 0.9734] | 52 |
| spread12 | sdf | 0.9750 ± 0.0045 | 0.9754 [0.9724, 0.9778] | 52 |
| spread12 | inr | 0.9538 ± 0.0070 | 0.9541 [0.9484, 0.9594] | 52 |
| spread12 | gaussian | 0.9211 ± 0.0126 | 0.9213 [0.9107, 0.9309] | 52 |
| clustered3 | nearest | 0.9224 ± 0.0182 | 0.9235 [0.9154, 0.9362] | 52 |
| clustered3 | sdf | 0.9232 ± 0.0182 | 0.9245 [0.9159, 0.9366] | 52 |
| clustered3 | inr | 0.8981 ± 0.0183 | 0.9022 [0.8847, 0.9099] | 52 |
| clustered3 | gaussian | 0.6174 ± 0.0280 | 0.6170 [0.5993, 0.6294] | 52 |

Primary comparison: SDF minus INR (positive favors SDF). Bootstrap intervals resample subjects; Holm correction covers five layouts.

| Layout | Paired mean difference | 95% bootstrap CI | Holm p | SDF higher / paired n |
| --- | ---: | --- | ---: | --- |
| spread2 | +0.0318 | [+0.0286, +0.0353] | 1.75e-09 | 52/52 |
| spread3 | +0.0320 | [+0.0293, +0.0346] | 1.75e-09 | 52/52 |
| spread6 | +0.0250 | [+0.0228, +0.0272] | 1.75e-09 | 52/52 |
| spread12 | +0.0212 | [+0.0197, +0.0228] | 1.75e-09 | 52/52 |
| clustered3 | +0.0251 | [+0.0232, +0.0270] | 1.75e-09 | 52/52 |

Coverage effect: spread3 minus clustered3 (positive favors spread views).

| Method | Paired mean difference | 95% bootstrap CI |
| --- | ---: | --- |
| nearest | +0.0285 | [+0.0249, +0.0324] |
| sdf | +0.0328 | [+0.0287, +0.0372] |
| inr | +0.0259 | [+0.0229, +0.0290] |
| gaussian | +0.2122 | [+0.2029, +0.2212] |

Original 34-view recipe reproduction: {'compared_runs': 312, 'max_absolute_dice_difference': 0.0}.

Source/run hashes are in worker config.json files. Counts 6 and 12 also expand the observed span. Scores use annotation x/y/angle coordinates and do not measure physical 3D reconstruction or clinical validity.
