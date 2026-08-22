# cardiac-reconstruction-evolved

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![TorchDiffEq](https://img.shields.io/badge/TorchDiffEq-Latest-orange.svg)

Sparse cardiac reconstruction with stabilized Gaussian occupancy fields, differentiable slice supervision, and mesh-based evaluation for interpretable 3D anatomy recovery.

This repository explores a Gaussian occupancy formulation for reconstructing cardiac anatomy from sparse imaging. Instead of representing the volume with a pure implicit MLP, it uses explicit 3D Gaussian primitives, a non-saturating occupancy rule, sparse-slice supervision, and marching-cubes mesh extraction for direct visual comparison against ground truth.

The current strongest result in this repo is a subject-specific reconstruction fit on the MITEA validation split. It is a working Gaussian-field reconstruction prototype with mesh-oriented diagnostics, not yet a mature multi-subject 4D Gaussian splatting system.

## What This Repo Does

- Represents anatomy as explicit 3D Gaussian occupancy kernels.
- Uses stabilized density aggregation: `occupancy(x) = 1 - exp(-density(x))`.
- Supports differentiable sparse-slice supervision and pose refinement in the encoder-based path.
- Includes a direct subject-fit pipeline that produces a usable mesh reconstruction.
- Publishes qualitative and quantitative diagnostics through the report and GitHub Pages site.

## Current Results

Current tracked subject-fit result from `runs/subject_fit_v01`:

- Sampled occupancy accuracy: `0.9655`
- Sampled IoU: `0.9503`
- Number of Gaussians: `1800`
- Fit steps: `600`

Artifacts in this repository include:

- Orthogonal slice comparison: [docs/assets/comparison_v01.png](docs/assets/comparison_v01.png)
- Reconstruction gallery: [docs/assets/visual_gallery.png](docs/assets/visual_gallery.png)
- Interactive mesh assets for the GitHub Pages site under [docs/assets](docs/assets)

## Main Workflows

### 1. Subject-Specific Reconstruction

This is the main working path in the repository right now.

```bash
python3 src_code/scripts/fit_subject_reconstruction.py \
  --data-dir /path/to/mitea \
  --run-dir runs/subject_fit_v01 \
  --split val \
  --sample-index 0 \
  --num-gaussians 1800 \
  --steps 600
```

### 2. 3D Evaluation

```bash
python3 src_code/scripts/evaluate_3d.py \
  --checkpoint runs/subject_fit_v01/subject_fit.pth
```

This writes a comparison figure and `metrics.json` beside the checkpoint output.
It also writes full-volume accuracy/IoU/Dice and affine-aware
`predicted_mesh.npz` and `target_mesh.npz` files. For a fresh checkpoint:

```bash
python3 src_code/scripts/evaluate_3d.py \
  --checkpoint runs/stabilization_v01/checkpoint_epoch_50.pth \
  --data-dir /path/to/mitea \
  --volume-resolution 64
```

### 3. Visual Gallery

```bash
python3 src_code/scripts/visual_gallery.py \
  --checkpoint runs/subject_fit_v01/subject_fit.pth \
  --output-path runs/subject_fit_v01/visual_gallery.png
```

### 4. Encoder-Based Training Path

The broader training path is still present for sparse-slice supervision experiments:

```bash
python3 src_code/scripts/train.py \
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

## Installation & Dependencies

Core dependencies used by the current pipeline:

- `torch`
- `torchvision`
- `torchdiffeq`
- `nibabel`
- `numpy`
- `matplotlib`

Some older or optional paths may still assume additional packages such as `pytorch3d`.

## Relation to Prior Project

This repository is distinct from the earlier [`cardiac-volume-reconstruction`](https://github.com/tushar-nayak/cardiac-volume-reconstruction) project.

The older project is the stronger baseline in terms of end-to-end experimental maturity. It focuses on reconstructing 3D cardiac volumes from sparse 2D echo views using coordinate-based implicit neural representations, transfer learning, and meta-learning for fast subject adaptation. Its central result is a multi-subject reconstruction pipeline with reported Dice and IoU summaries across saved experiments.

This repository explores a different modeling direction. Instead of representing anatomy with an MLP-based implicit field, it uses explicit Gaussian occupancy primitives, differentiable slice rendering, and diagnostic tooling built around occupancy stability, mesh extraction, and visual inspection. In practice, this makes `cardiac-reconstruction-evolved` more of a Gaussian-field reconstruction prototype and experimentation branch, whereas `cardiac-volume-reconstruction` is the more complete INR-based cardiac reconstruction project.

In short:

- `cardiac-volume-reconstruction`: established INR / meta-learning cardiac reconstruction pipeline with broader evaluation.
- `cardiac-reconstruction-evolved`: experimental Gaussian occupancy reconstruction path focused on explicit primitives, stabilization, and mesh-oriented diagnostics.
