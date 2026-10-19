# Project Tracker: cardiac-reconstruction

## Timeline

### Phase 1: Foundation & Data (Days 1-2) - **DONE**
- [x] Initialize project structure (`code/`, `runs/`, `data/`).
- [x] Implement `MITEADataset` loader.
- [x] Basic Neural ODE and Gaussian Model architecture.

### Phase 2: Refinement & Smoke Run (Days 3-4) - **DONE**
- [x] **Smoke Run:** Verify the end-to-end pipeline with a small subset of data.
- [x] **Loss Function:** Implement proper 3D occupancy supervision using affine transforms.
- [x] **Git Sync:** Commit and push the initial working skeleton.

### Phase 3: Advanced Modeling (Days 5-9) - **DONE**
- [x] **Radiological Rasterizer:** Integrate or simulate the density accumulation rendering.
- [x] **Pose Optimization:** Implement differentiable slice alignment.
- [x] **VLM Integration:** Add semantic embedding support for queryable regions.

### Phase 4: Validation & Visualization (Days 10-14) - **DONE**
- [x] **Evaluation:** Compare against simple persistence and linear interpolation baselines.
- [x] **Visualization:** Create 4D animations of the contracting ventricle.
- [x] **Report:** Document findings and performance metrics.

---

## Status Notes
- **Current State:** Stabilized occupancy pipeline, updated evaluation scripts, and current diagnostics in `runs/`.
- **Next Step:** Improve cavity/wall separation and validate the latest evaluation outputs.
