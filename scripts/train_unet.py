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

from cardiac_common.cohort import LV_CAVITY, enumerate_volumes, load_label
from cardiac_common.population import listed_training_subjects, training_subjects
from cardiac_common.apical import training_observer
from cardiac_common.unet import train_unet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--manifest", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--train-folds", nargs="+", type=int)
    source.add_argument("--train-subjects-file", default=None,
                        help="JSON file with a \"subjects\" list of development subjects")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--checkpoints", nargs="+", type=int, required=True,
                        help="Steps at which to save an artifact (must include --iterations if wanted)")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target", choices=("union", "cavity"), default="union",
                        help="Label union (protocol v1/v2) or LV cavity only")
    parser.add_argument("--phases", nargs="+", choices=("ED", "ES"), default=["ED"],
                        help="Scan-1 phases used as training shapes")
    parser.add_argument("--observation", choices=("protocol", "apical"), default="protocol",
                        help="Protocol-v2 layouts, or misplaced apical views (cavity target)")
    args = parser.parse_args()
    if not all(0 < step <= args.iterations for step in args.checkpoints):
        parser.error("checkpoints must lie in (0, iterations]")
    outputs = {step: Path(f"{args.output_prefix}_it{step}_seed{args.seed}.pt") for step in args.checkpoints}
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(path)
    manifest = json.loads(Path(args.manifest).read_text())
    if args.train_folds is not None:
        subjects = training_subjects(manifest, args.train_folds)
    else:
        listed = json.loads(Path(args.train_subjects_file).read_text())["subjects"]
        subjects = listed_training_subjects(manifest, listed)
    paths = {(v.subject, v.scan, v.phase): v.label_path for v in enumerate_volumes(args.data_dir)}
    loaded = [load_label(paths[(subject, 1, phase)]) for subject in subjects for phase in args.phases]
    masks = [label == LV_CAVITY if args.target == "cavity" else label > 0 for label, _ in loaded]
    observe_fn = None
    if args.observation == "apical":
        if args.target != "cavity":
            parser.error("--observation apical requires --target cavity")
        masks, observe_fn = training_observer([label for label, _ in loaded], [a for _, a in loaded])
    del loaded
    source = args.train_subjects_file or f"folds {args.train_folds}"
    print(f"training on {len(subjects)} subjects from {source}", flush=True)
    _, history, seconds, saved = train_unet(masks, args.iterations, args.lr, args.seed, args.device,
                                            snapshots=set(args.checkpoints), observe_fn=observe_fn)
    for step, path in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"method": "unet", "state_dict": saved[step], "training_subjects": subjects,
                    "train_folds": args.train_folds, "subject_list": args.train_subjects_file,
                    "target": args.target, "phases": args.phases, "observation": args.observation,
                    "manifest_sha256": manifest["splits_sha256"],
                    "lr": args.lr, "iterations": step, "seed": args.seed,
                    "loss_history_per_500": history[:step // 500], "train_seconds_total_run": seconds},
                   path)
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
