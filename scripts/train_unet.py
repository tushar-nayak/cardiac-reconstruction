"""Train the protocol-v2 completion U-Net on development folds.

Writes one artifact per requested checkpoint, named
``<output-prefix>_it<steps>_seed<seed>.pt``. Example (fold-0 development):
    .venv/bin/python scripts/train_unet.py --data-dir dataset/mitea \
        --manifest benchmarks/splits/mitea_cohort_v1.json --train-folds 1 2 3 4 \
        --lr 1e-3 --iterations 8000 --checkpoints 4000 8000 --seed 17 \
        --output-prefix runs/protocol_v2_models/unet_train1234_lr1e-3
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cardiac_common.cohort import enumerate_volumes, load_label
from cardiac_common.population import training_subjects
from cardiac_common.unet import train_unet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train-folds", nargs="+", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--checkpoints", nargs="+", type=int, required=True,
                        help="Steps at which to save an artifact (must include --iterations if wanted)")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not all(0 < step <= args.iterations for step in args.checkpoints):
        parser.error("checkpoints must lie in (0, iterations]")
    outputs = {step: Path(f"{args.output_prefix}_it{step}_seed{args.seed}.pt") for step in args.checkpoints}
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(path)
    manifest = json.loads(Path(args.manifest).read_text())
    subjects = training_subjects(manifest, args.train_folds)
    paths = {(v.subject, v.scan, v.phase): v.label_path for v in enumerate_volumes(args.data_dir)}
    masks = [load_label(paths[(subject, 1, "ED")])[0] > 0 for subject in subjects]
    print(f"training on {len(subjects)} subjects from folds {args.train_folds}", flush=True)
    _, history, seconds, saved = train_unet(masks, args.iterations, args.lr, args.seed, args.device,
                                            snapshots=set(args.checkpoints))
    for step, path in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"method": "unet", "state_dict": saved[step], "training_subjects": subjects,
                    "train_folds": args.train_folds, "manifest_sha256": manifest["splits_sha256"],
                    "lr": args.lr, "iterations": step, "seed": args.seed,
                    "loss_history_per_500": history[:step // 500], "train_seconds_total_run": seconds},
                   path)
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
