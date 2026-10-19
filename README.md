# HeartMaker: Cardiac Reconstruction with Gaussian Occupancy and Implicit Neural Representations

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![TorchDiffEq](https://img.shields.io/badge/TorchDiffEq-Latest-orange.svg)

HeartMaker is the central project for exploring two cardiac reconstruction approaches: explicit Gaussian occupancy fields and coordinate-based implicit neural representations (INRs).

The methods live side by side in `code_gaussian/` and `code_inr/`, with shared MITEA data loading and comparable reconstruction metrics. The website starts with INR selected and keeps the two results pages separate.

The Gaussian method uses explicit 3D primitives and a non-saturating occupancy rule. The INR method represents occupancy as a coordinate MLP and fits it from sparse annotated planes. Their existing results use different supervision protocols, so the reported scores should be read in their own experiment context.

## What This Repo Does

- Represents anatomy as explicit 3D Gaussian occupancy kernels.
- Represents anatomy as a continuous implicit neural occupancy field.
- Uses stabilized density aggregation: `occupancy(x) = 1 - exp(-density(x))`.
- Supports differentiable sparse-slice supervision and pose refinement in the encoder-based path.
- Includes a direct subject-fit pipeline that produces a usable mesh reconstruction.
- Publishes separate results pages for each approach through the [HeartMaker project site](https://tushar-nayak.github.io/heart-maker/).
- Includes a local copy of the INR project's interactive results site, reports, and viewer exports, so those visuals do not depend on the original repository being online.

## Current Results

### Previous Stable Recipe

The main subject-fitting path now defaults to the May 15 recipe: 1,800
localized, physical-space Gaussians initialized and optimized using the full
3D label. The historical split selects `MITEA_111` at validation index 0.
These are full-label representation fits, distinct from the sparse-plane pilot
and the older encoder experiment.

| Subject | Steps | Native-grid Dice | Native-grid IoU |
| --- | ---: | ---: | ---: |
| MITEA_111 | 600 | 0.9449 | 0.8956 |
| MITEA_111 | 1800 | 0.9776 | 0.9562 |
| MITEA_071 | 600 | 0.9678 | 0.9377 |
| MITEA_071 | 1800 | 0.9812 | 0.9632 |

The 600-step reproduction's historical sampled-point IoU is 0.9487 on
`MITEA_111`, close to the archived 0.9503. Sampled and whole-grid metrics are
reported separately. These two subjects use seed 17; they do not establish
sparse-view performance or population-level generalization.
See [the restoration notes](benchmarks/may2026_gaussian_reference.md) for
commands, geometry diagnostics, and the remaining limitations.

### Earlier encoder experiment

Current best completed encoder result from `runs/ablation_neg4`:

- Full-volume accuracy: `0.9510`
- Full-volume IoU: `0.4519`
- Full-volume Dice: `0.6225`
- Number of Gaussians: `1000`
- Training epochs: `40`
- Evaluation split/sample: `val/0`

Negative-loss ablation on the same validation sample:

| Negative-loss weight | Full-volume IoU | Full-volume Dice |
| ---: | ---: | ---: |
| 4 | 0.4519 | 0.6225 |
| 8 | 0.4458 | 0.6167 |
| 16 | 0.3778 | 0.5484 |

The weight-4 run is the best completed ablation recorded here. No completed
2,000-Gaussian result is included in these reported results.

Historical encoder assets are retained in
[the research archive](benchmarks/legacy_encoder_assets/) outside the published
website. Current interactive Gaussian meshes and slice data live under
[docs/assets/gaussian-reference](docs/assets/gaussian-reference).

## Main Workflows

### Shared sparse-plane benchmark

Run both methods with the same annotated mask planes, minibatch sequence, BCE
loss, and number of optimizer steps:

```bash
.venv/bin/python scripts/benchmark_sparse.py \
  --data-dir dataset/mitea \
  --run-dir runs/shared_sparse_baseline_v01 \
  --plane-layout orthogonal \
  --steps 1000 \
  --save-occupancy \
  --device cuda
```

By default this fits the first validation subject with both methods. The three
central orthogonal planes are selected from volume dimensions alone. Training
reads labels on those planes; default Gaussian initialization uses uniformly sampled
positions without consulting the 3D target. Both fields query normalized XYZ
coordinates. This is reconstruction from sparse binary annotations; an
echo-image encoder is a separate next step. Matching update counts does not
match model size or runtime.

Each new run directory contains the subject split manifest, configuration,
per-method checkpoints, and `results.json`. Scores include whole-volume Dice
and IoU, overlap on unobserved voxels, physical volume error, fitting/evaluation
time, parameter count, and peak allocated CUDA memory. A supervision fingerprint
records that the methods received the same observed labels. Existing run
directories are rejected to prevent accidental overwrites.

Evaluation also reports **ΔK**, the signed difference between predicted and
target integrated Gaussian curvature, and its absolute value. Both masks are
meshed at 0.5 after thresholding the prediction at the configured probability
threshold. Triangle angle defects are measured on meshes transformed by the
NIfTI affine. Reports include surface area, connected components, Euler
characteristic, boundary edges, nonmanifold edges/vertices, and a numerical
Gauss–Bonnet residual. On closed manifold surfaces, the integral equals
`2π × Euler characteristic`; zero ΔK does not establish anatomical accuracy.
The integral is dimensionless; area assumes the affine uses millimeters.

For open, missing, degenerate, or nonmanifold surfaces, ΔK is `null` with a
status explaining why the closed-surface comparison is unavailable. Open
manifolds report interior angle defects and boundary turning separately. Meshes
are not padded or capped at the grid boundary. The checks concern mesh
connectivity and face degeneracy; they do not detect geometric self-intersections.
Summary statistics count valid and unavailable ΔK runs separately. These
diagnostics are evaluation outputs and do not change either training loss.

To evaluate an existing saved occupancy without retraining:

```bash
.venv/bin/python scripts/evaluate_geometry.py \
  --prediction runs/shared_sparse_baseline_v01/inr/MITEA_071/seed_17/predicted_occupancy.pt \
  --target dataset/mitea/labels/MITEA_071_scan1_ED.nii.gz \
  --output runs/shared_sparse_baseline_v01/inr/MITEA_071/seed_17/geometry.json
```

Use the Gaussian run's occupancy path to evaluate that method. Match
`--eval-stride` to the saved run when evaluating subsampled outputs. Each output
file must be new; existing files are rejected.

Use `--sample-indices 0 1 2 --seeds 17 23 42` for multiple subjects and seeds,
or `--all-subjects` for the selected split. The current loader covers scan 1 ED
subjects. `--plane-layout axial --num-planes 3` selects evenly spaced axial
planes without inspecting foreground. `--eval-stride 8 --steps 5` provides a
quick smoke check; coarse-grid overlap and volume estimates from that mode
must not be reported as native-resolution results. Surface-distance metrics
and broader scan/phase coverage are still future work.

The first single-subject run and its exact settings are recorded in
[the initial benchmark notes](benchmarks/sparse_planes_v01.md).

Gaussian optimization experiments can use `--methods gaussian`,
`--gaussian-init observed`, and `--lr 0.003`. The optional initialization seeds centers from foreground
coordinates on the selected observation planes, with small spatial jitter. It
does not inspect labels elsewhere in the volume. If the observed masks contain
no foreground, it retains uniform initialization. The chosen initialization,
learning rate, and step count are recorded in each result. Uniform initialization
and the original learning rate remain the defaults for reproducing the baseline.
The [Gaussian optimization pilot](benchmarks/gaussian_optimization_v01.md)
records the ablations, matched three-subject/three-seed results, and commands.

### 1. Subject-Specific Reconstruction

This is the main working path in the repository right now.

```bash
python3 code_gaussian/scripts/fit_subject_reconstruction.py \
  --data-dir /path/to/mitea \
  --run-dir runs/gaussian_may2026_reference_v01 \
  --recipe may2026 \
  --split val \
  --sample-index 0 \
  --num-gaussians 1800 \
  --steps 600 \
  --evaluate
```

Use a new run directory each time. `--evaluate` saves the native-resolution
occupancy tensor, whole-grid metrics, ΔK geometry checks, and the separately
labeled historical sampled-point diagnostic. Use `--steps 1800` to reproduce the
refined fit. The May recipe uses the historical sorted split by default;
`--split-policy shuffled` selects the current subject split instead. Checkpoints
record subject identity and split policy so evaluation cannot silently select a
different patient. `--recipe stabilized` retains the later centroid-weighted
sampling path and defaults to the shuffled split.

### 2. 3D Evaluation

```bash
python3 code_gaussian/scripts/evaluate_3d.py \
  --checkpoint runs/ablation_neg4/checkpoint_epoch_40.pth \
  --data-dir /path/to/mitea
```

This writes a comparison figure and `metrics.json` beside the checkpoint output.
It also writes full-volume accuracy/IoU/Dice and affine-aware
`predicted_mesh.npz` and `target_mesh.npz` files. For a fresh checkpoint:

```bash
python3 code_gaussian/scripts/evaluate_3d.py \
  --checkpoint runs/stabilization_v01/checkpoint_epoch_50.pth \
  --data-dir /path/to/mitea \
  --volume-resolution 64
```

### 3. Implicit neural representation baseline

The repository also includes a coordinate based implicit neural representation
(INR), adapted from the approach in
[`cardiac-volume-reconstruction`](https://github.com/tushar-nayak/cardiac-volume-reconstruction).
It learns a continuous occupancy field from a few annotated axial planes, then
queries that field over the complete volume. This first local baseline uses
fixed, stratified planes and per-subject fitting; the upstream shared-prior,
pose-learning, and Reptile meta-learning experiments are not included yet.

The supplied archive expands to `dataset/mitea/`:

```bash
python3 -m zipfile -e dataset/cap-mitea.zip dataset
python3 code_inr/fit_subject_inr.py \
  --data-dir dataset/mitea \
  --run-dir runs/inr_subject_v01 \
  --split val \
  --sample-index 0 \
  --num-slices 3
```

The INR model and fitting code live in `code_inr/`. Its loader and common metric
utilities are imported from `code_gaussian/`. The checkpoint, predicted
occupancy, and full-volume metrics are saved under the run directory. Gaussian
models and workflows live in `code_gaussian/`.

### 4. Visual Gallery

```bash
python3 code_gaussian/scripts/visual_gallery.py \
  --checkpoint runs/ablation_neg4/checkpoint_epoch_40.pth \
  --data-dir /path/to/mitea \
  --output-path runs/ablation_neg4/visual_gallery.png
```

### 5. Encoder-Based Training Path

The broader training path is still present for sparse-slice supervision experiments:

```bash
python3 code_gaussian/scripts/train.py \
  --data-dir /path/to/mitea \
  --run-dir runs/stabilization_v01 \
  --epochs 50
```

This path includes the encoder, Gaussian generator, rasterizer, and pose optimizer, but it is not the strongest current result compared with the direct subject-fit workflow.
It now trains an ED-to-ES latent ODE and stores its state in the checkpoint.
Use `--num-sparse-slices 1`, `2`, or `3` to control the input planes.

## Project Status

- `Working now`: static subject-specific Gaussian occupancy reconstruction with mesh extraction.
- `Partially explored`: differentiable sparse-slice supervision and pose refinement.
- `Experimental`: Neural ODE dynamics and 4D animation tooling.
- `Not current mainline`: VLM-driven semantic tracking and dense multi-view 3DGS claims.

The original pre-overhaul implementation is preserved on the pushed `legacy`
branch. The active `main` branch now uses a shared occupancy implementation,
reproducible subject splits, seeded fitting, a maintained evaluation entry
point, and standard-library tests. Dataset paths are required explicitly so
results do not depend on the developer machine that created the checkpoint.

Run the lightweight checks with:

```bash
python3 -m unittest discover -s tests -v
```

The MITEA setup currently used here is best suited for static 3D reconstruction and limited ED/ES experiments, not for a full dense multi-view RGB 3DGS pipeline.

## Documentation

- Technical report: [report.md](report.md)
- GitHub Pages site: [docs/index.html](docs/index.html)
- Gaussian results page: [docs/gaussian.html](docs/gaussian.html)
- Previous Stable Recipe and refined fits: [docs/gaussian.html](docs/gaussian.html)
- INR results page: [docs/inr.html](docs/inr.html)

## Installation & Dependencies

Create the repository-local environment and install the shared dependencies:

```bash
conda create --prefix .venv --override-channels --channel conda-forge python=3.12 pip
conda activate ./.venv
python -m pip install -r code_gaussian/requirements.txt
```

You can also call the environment interpreter directly as `.venv/bin/python`.

Core dependencies used by the current pipeline:

- `torch`
- `torchvision`
- `torchdiffeq`
- `nibabel`
- `numpy`
- `matplotlib`

Some older or optional paths may still assume additional packages such as `pytorch3d`.

## INR Results Archive

The published INR results and interactive viewers are preserved locally under [`docs/inr-archive/`](docs/inr-archive/). The six-scan gallery includes selectable 2D slice and 3D mesh viewers; ten additional comparison/overlay viewers and the original result pages are included as well. HeartMaker’s [INR results page](https://tushar-nayak.github.io/heart-maker/inr.html) embeds the local gallery.

- [Standalone local results site](docs/inr-archive/index.html)
- [Interactive six-scan gallery](docs/inr-archive/gallery.html)
- [Results index](docs/inr-archive/results_index.html) · [Report](docs/inr-archive/report.html)
- [Original project site](https://tushar-nayak.github.io/cardiac-volume-reconstruction/) · [Source repository](https://github.com/tushar-nayak/cardiac-volume-reconstruction)

The local INR baseline currently ports the coordinate MLP with positional encoding and sparse-plane occupancy supervision. The upstream shared-prior, pose-learning, and Reptile meta-learning workflows have not yet been migrated into `code_inr/`.

The current Gaussian subject-fit script uses sampled points throughout the 3D label, while the local INR baseline sees only selected planes. Their metrics are therefore not an equal-supervision comparison.
