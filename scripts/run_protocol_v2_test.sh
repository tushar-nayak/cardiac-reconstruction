#!/usr/bin/env bash
# One-time final test evaluation for the frozen protocol-v2 settings.
# Run from the repository root after verifying that the project GPU is idle.
set -euo pipefail

if command -v nvidia-smi >/dev/null 2>&1; then
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1)
  if (( free_mb < 7000 )); then
    echo "Refusing to start: only ${free_mb} MiB GPU memory is free; wait for other workloads to finish." >&2
    exit 2
  fi
fi

PYTHON=.venv/bin/python
DATA=dataset/mitea
MANIFEST=benchmarks/splits/mitea_cohort_v1.json
MODEL_DIR=runs/protocol_v2_models
SSM_MODEL="$MODEL_DIR/ssm_final_train_folds01234.pt"
UNET_PREFIX="$MODEL_DIR/unet_final_train_folds01234_lr3e-4"
UNET_MODELS=(
  "${UNET_PREFIX}_it4000_seed17.pt"
  "${UNET_PREFIX}_it4000_seed23.pt"
  "${UNET_PREFIX}_it4000_seed42.pt"
)
for path in runs/protocol_v2_test_20261022_orthogonal \
            runs/protocol_v2_test_20261022_axial \
            runs/protocol_v2_test_20261022_report.json; do
  if [[ -e "$path" ]]; then
    echo "Refusing to overwrite existing final-evaluation output: $path" >&2
    exit 2
  fi
done

if [[ ! -f "$SSM_MODEL" ]]; then
  "$PYTHON" scripts/train_ssm.py --data-dir "$DATA" --manifest "$MANIFEST" \
    --train-folds 0 1 2 3 4 --max-modes 40 --output "$SSM_MODEL"
fi

for seed in 17 23 42; do
  artifact="${UNET_PREFIX}_it4000_seed${seed}.pt"
  if [[ ! -f "$artifact" ]]; then
    "$PYTHON" scripts/train_unet.py --data-dir "$DATA" --manifest "$MANIFEST" \
      --train-folds 0 1 2 3 4 --lr 3e-4 --iterations 4000 --checkpoints 4000 \
      --seed "$seed" --output-prefix "$UNET_PREFIX"
  fi
done

for layout in orthogonal axial; do
  run_dir="runs/protocol_v2_test_20261022_${layout}"
  layout_args=(--plane-layout "$layout")
  if [[ "$layout" == axial ]]; then
    layout_args+=(--axial-fractions 0.375 0.5 0.625)
  fi
  "$PYTHON" scripts/benchmark_sparse.py --data-dir "$DATA" --manifest "$MANIFEST" \
    --subset test --allow-test-evaluation --run-dir "$run_dir" \
    --methods inr gaussian interpolation rbf ssm unet gp prior_inr \
    --seeds 17 23 42 --steps 1000 --batch-size 4096 --lr 1e-3 \
    --hidden-dim 64 --layers 4 --frequencies 4 --num-gaussians 256 \
    --gaussian-init uniform --rbf-kernel r3 --rbf-smoothing 0.01 \
    --ssm-model "$SSM_MODEL" --ssm-modes 20 --ssm-prior 0.001 \
    --unet-models "${UNET_MODELS[@]}" --gp-length 20 --gp-prior-mean -10 \
    --prior-inr-lr 1e-4 --prior-inr-weight 0 \
    --save-occupancy-subjects MITEA_004 MITEA_017 --device cuda "${layout_args[@]}"
done

"$PYTHON" scripts/summarize_benchmark.py report --family v2 --expect-subset test \
  --manifest "$MANIFEST" \
  --runs runs/protocol_v2_test_20261022_orthogonal \
         runs/protocol_v2_test_20261022_axial \
  --output runs/protocol_v2_test_20261022_report.json
