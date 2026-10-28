"""Execute the frozen interpolation/count/coverage protocol without tuning."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cardiac_common.radial import (
    COMMON_VIEWS, LAYOUTS, METHODS, SEEDS, RadialInterpolator, radial_metrics,
)
from scripts.evaluate_feecho4d import (
    all_grid_coordinates, fit_one, parse_cfg, predict_gaussian,
)

QUAL_SUBJECTS = ("Patient001", "Patient016")
QUAL_QUERIES = (2, 14, 24, 36)  # 5, 65, 115, 175 degrees


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_patient(root, patient):
    folder = root / "FeEcho4D_Annotated" / patient
    cfg = parse_cfg((folder / f"{patient}_info.cfg").read_text())
    frame = int(cfg["ed_time"])
    masks, raw_png = [], []
    for view in range(1, 38):
        name = folder / "mask" / f"{patient}_slice{view:03d}time{frame:03d}.png"
        raw = name.read_bytes()
        with Image.open(io.BytesIO(raw)) as image:
            values = np.array(image)
        if values.ndim != 2 or not set(np.unique(values)) <= {0, 127, 255}:
            raise ValueError(f"unexpected mask format: {name}")
        masks.append(values > 0)
        raw_png.append(raw)
    return np.stack(masks), cfg, raw_png


def predict_field(model, method, shape, query_ids, device):
    points = all_grid_coordinates(shape, query_ids, device)
    with torch.inference_mode():
        if method == "gaussian":
            probability = predict_gaussian(model, points)
        else:
            chunks = [model(points[start:start + 32768]).reshape(-1).cpu()
                      for start in range(0, len(points), 32768)]
            probability = torch.cat(chunks)
    if not torch.isfinite(probability).all():
        raise RuntimeError("nonfinite query predictions")
    return (probability.numpy() >= 0.5).reshape(len(query_ids), *shape[1:])


def configuration(args, subjects):
    files = ["benchmarks/feecho4d_ablation_v1.md", "cardiac_common/radial.py",
             "scripts/run_feecho4d_ablation.py", "scripts/evaluate_feecho4d.py",
             "code_inr/inr_model.py", "code_gaussian/sparse_model.py",
             "code_gaussian/src/occupancy.py", "cardiac_common/interpolation.py"]
    return {
        "protocol": files[0], "source_sha256": {f: sha256((ROOT / f).read_bytes()) for f in files},
        "subjects": subjects, "layouts": {key: list(LAYOUTS[key]) for key in args.layouts},
        "methods": args.methods, "seeds": list(SEEDS), "common_views": list(COMMON_VIEWS),
        "data_root": str(args.data_root.resolve()), "device": args.device,
        "steps": 1000, "batch_size": 4096, "lr": 0.001, "threshold": 0.5,
        "coordinates": "normalized pixel x/y and angle; not physical XYZ",
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "execution": "parallel GPU workers permitted; wall times include contention",
    }


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    root = args.data_root
    if (root / "FeEcho4D").is_dir():
        root = root / "FeEcho4D"
    subjects = sorted(p.name for p in (root / "FeEcho4D_Annotated").glob("Patient[0-9][0-9][0-9]")
                      if p.is_dir())
    if len(subjects) != 52:
        raise ValueError(f"expected 52 annotated subject folders, found {len(subjects)}")
    if args.subjects:
        if set(args.subjects) - set(subjects):
            raise ValueError("unknown subject selection")
        subjects = [s for s in subjects if s in args.subjects]
    config = configuration(args, subjects)
    config_path = args.output_dir / "config.json"
    if args.output_dir.exists():
        if not args.resume:
            raise FileExistsError(f"output exists: {args.output_dir}; use explicit --resume")
        if json.loads(config_path.read_text()) != config:
            raise ValueError("resume configuration/source hashes do not match")
    else:
        args.output_dir.mkdir(parents=True)
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    record_path = args.output_dir / "runs.jsonl"
    previous = [json.loads(line) for line in record_path.read_text().splitlines()] if record_path.exists() else []
    done = {(r["subject"], r["layout"], r["method"], r["seed"]) for r in previous}
    if len(done) != len(previous):
        raise ValueError("duplicate previous records")
    with record_path.open("a", buffering=1) as handle:
        for index, subject in enumerate(subjects, 1):
            print(f"[{index}/{len(subjects)}] {subject}", flush=True)
            masks, cfg, pngs = load_patient(root, subject)
            for layout in args.layouts:
                observed_ids = LAYOUTS[layout]
                observed = masks[np.array(observed_ids) - 1].copy()
                query_ids = [i for i in range(1, 38) if i not in observed_ids]
                obs_hash = sha256(json.dumps({"shape": masks.shape, "views": observed_ids}).encode()
                                  + observed.tobytes())
                eval_hash = sha256(json.dumps({"shape": masks.shape, "common_views": COMMON_VIEWS}).encode())
                original_hash = sha256(b"".join(pngs[i - 1] for i in observed_ids))
                points, labels = None, None
                for method in args.methods:
                    seeds = SEEDS if method in ("inr", "gaussian") else (None,)
                    for seed in seeds:
                        key = (subject, layout, method, seed)
                        if key in done:
                            continue
                        row = {"subject": subject, "diagnosis": cfg["diagnosis"],
                               "ed_time": int(cfg["ed_time"]), "layout": layout,
                               "observed_views": list(observed_ids), "method": method, "seed": seed,
                               "observation_sha256": obs_hash, "evaluation_sha256": eval_hash,
                               "observed_png_sha256": original_hash, "status": "failed"}
                        model = None
                        try:
                            if method in ("nearest", "sdf"):
                                start = time.perf_counter()
                                interpolator = RadialInterpolator(observed_ids, observed, method)
                                row["fit_seconds"] = time.perf_counter() - start
                                start = time.perf_counter()
                                pred = interpolator.predict(query_ids)
                            else:
                                if points is None:
                                    points = all_grid_coordinates(masks.shape, list(observed_ids), device)
                                    labels = torch.from_numpy(observed.astype(np.float32).reshape(-1))
                                model, fit_seconds, final_bce = fit_one(method, points, labels, None, device, seed)
                                row.update(fit_seconds=fit_seconds, final_bce=final_bce)
                                start = time.perf_counter()
                                pred = predict_field(model, method, masks.shape, query_ids, device)
                            row["query_seconds"] = time.perf_counter() - start
                            # Only scoring below receives the withheld target masks.
                            row["metrics"] = radial_metrics(pred, masks[np.array(query_ids) - 1],
                                                            query_ids, observed_ids)
                            row["empty_prediction"] = bool(not pred.any())
                            row["status"] = "ok"
                            if subject in QUAL_SUBJECTS and seed in (None, 17):
                                positions = [query_ids.index(i) for i in QUAL_QUERIES]
                                qual = args.output_dir / "qualitative"
                                qual.mkdir(exist_ok=True)
                                name = qual / f"{subject}_{layout}_{method}.npz"
                                np.savez_compressed(name, predicted=pred[positions],
                                                    target=masks[np.array(QUAL_QUERIES) - 1],
                                                    query_ids=np.array(QUAL_QUERIES))
                        except Exception as exc:
                            row["error"] = f"{type(exc).__name__}: {exc}"
                            print(f"FAIL {key}: {row['error']}", flush=True)
                        finally:
                            if model is not None:
                                del model
                        handle.write(json.dumps(row, allow_nan=False) + "\n")
                        done.add(key)
                del points, labels
    print(f"Completed {len(done)} records in {args.output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "dataset/FeEcho4D_extracted")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layouts", choices=list(LAYOUTS), nargs="+", default=list(LAYOUTS))
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if len(set(args.layouts)) != len(args.layouts) or len(set(args.methods)) != len(args.methods):
        parser.error("duplicate layouts/methods")
    run(args)


if __name__ == "__main__":
    main()
