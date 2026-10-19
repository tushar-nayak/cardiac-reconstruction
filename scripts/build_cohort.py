"""Write the frozen subject-level split manifest for MITEA experiments."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cardiac_common.cohort import build_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="benchmarks/splits/mitea_cohort_v1.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error(f"{output} exists; the split is frozen once written")
    manifest = build_manifest(args.data_dir, args.folds, args.test_fraction, args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    splits = manifest["splits"]
    print(f"test={len(splits['test'])} folds={[len(f) for f in splits['folds']]} "
          f"sha256={manifest['splits_sha256'][:12]}")


if __name__ == "__main__":
    main()
