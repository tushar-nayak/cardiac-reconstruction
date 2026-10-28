# MITEA NeRF rendering pilot

This is a renderer and data-pipeline pilot on `MITEA_007`, a development-fold-0
subject. It is **not** a sparse-plane cardiac reconstruction result.

The MITEA inputs available here are reconstructed 3D ultrasound intensity
volumes with aligned labels, not calibrated collections of conventional
NeRF camera images. To exercise volumetric rendering without misrepresenting
the acquisition, the pilot forms synthetic orthographic views by rendering
the full volume with label-derived density and ultrasound intensity as
grayscale radiance. The field is trained on the `+x`, `+y`, and `+z` rendered
views and checked on reverse directions. Full-volume data generate both the
training targets and the held-out rendering targets; consequently these
measure renderer behavior only, not recovery from sparse clinical views.

The model predicts nonnegative density and grayscale radiance from encoded
position and view direction. For samples along a ray, it uses standard
emission-absorption weights

\[
  \alpha_i=1-\exp(-\sigma_i\delta_i),\qquad
  w_i=\alpha_i\prod_{j<i}(1-\alpha_j),\qquad
  \widehat C=\sum_iw_i c_i.
\]

An auxiliary rendered-opacity loss stabilizes the label-derived density.
There is no test-subject access. Pilot outputs are written under the
git-ignored `runs/nerf_mitea_pilot/` directory.

## Initial run

The default run completed 1,200 optimization steps on CUDA in 16.0 seconds.
Training loss fell from 0.2268 to 0.00245. Synthetic-view intensity PSNR was
29.9--32.0 dB on the three training directions and 21.4--23.7 dB on the
reverse-direction checks. Reverse-view opacity MSE was 0.0055--0.0582,
indicating that density recovery is not yet consistent across unseen
directions. These are per-subject synthetic-render diagnostics, not MITEA
sparse-plane Dice or evidence of clinical NeRF performance.

Run with:

```bash
./.venv/bin/python scripts/train_nerf_mitea.py
```

Before this can be called sparse-view reconstruction, a follow-up must define
an acquisition-faithful image formation model and train only from observed
planes while evaluating the untouched target volume under the frozen split.
