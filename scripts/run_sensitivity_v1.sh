#!/usr/bin/env bash
# Sensitivity analyses v1 (benchmarks/mitea_sensitivity_v1.md): development folds 1-4 only; the
# held-out test set is never selected. Run from the repository root, for example in tmux:
#   tmux new-session -d -s sensitivity_v1 \
#     'scripts/run_sensitivity_v1.sh 2>&1 | tee -a runs/mitea_sensitivity_v1/run.log'
# Steps run one at a time. Finished units are skipped, so rerunning resumes; a partial unit stops
# the run for inspection. DRY_RUN=1 prints the commands without running them.
set -euo pipefail

PYTHON=.venv/bin/python
DATA=dataset/mitea
MANIFEST=benchmarks/splits/mitea_cohort_v1.json
ROOT=runs/mitea_sensitivity_v1
MODELS=$ROOT/models
SUBSETS=$ROOT/subsets
LOGS=$ROOT/logs
FOLDS=(1 2 3 4)
DRY_RUN=${DRY_RUN:-0}

# Frozen v2 settings; seed 17 only; no GP uncertainty summaries.
COMMON=(--data-dir "$DATA" --manifest "$MANIFEST" --protocol-name mitea_sensitivity_v1 --seeds 17
        --steps 1000 --batch-size 4096 --lr 1e-3 --hidden-dim 64 --layers 4 --frequencies 4
        --rbf-kernel r3 --rbf-smoothing 0.01 --ssm-prior 0.001 --gp-length 20 --gp-prior-mean -10
        --prior-inr-lr 1e-4 --prior-inr-weight 0 --no-gp-uncertainty --device cuda)
ALL=(inr interpolation rbf ssm unet gp prior_inr)
PRIORS=(ssm prior_inr unet mean_shape)
declare -A CONDITION=([clean]="" [random2]="--section-noise-sd-mm 2" [random4]="--section-noise-sd-mm 4"
                      [inside2]="--section-offset-mm -2" [outside2]="--section-offset-mm 2")

log() { echo "[$(date '+%F %T')] $*"; }

execute() {  # LOGFILE COMMAND...: run into LOGFILE, or print the command under DRY_RUN=1
  local output=$1; shift
  if [[ $DRY_RUN == 1 ]]; then echo "  $* > $output"; return 0; fi
  "$@" > "$output" 2>&1
}

other_folds() {
  local fold
  for fold in 0 1 2 3 4; do if [[ $fold != "$1" ]]; then printf '%s ' "$fold"; fi; done
}

subjects_of() {
  "$PYTHON" -c 'import json, sys; print(" ".join(json.load(open(sys.argv[1]))["subjects"]))' "$1"
}

fractions() {  # parallel sections at a spacing of 1/8 of the depth, centred on 0.5
  "$PYTHON" -c 'import sys; n = int(sys.argv[1]); print(" ".join(str(0.5 + (i - (n - 1) / 2) / 8) for i in range(n)))' "$1"
}

priors() {  # NAME SOURCE-ARGS...: SSM and seed-17 U-Net trained on the same subjects
  local name=$1; shift
  local ssm=$MODELS/ssm_$name.pt prefix=$MODELS/unet_${name}_lr3e-4
  if [[ ! -f $ssm ]]; then
    log "build $ssm"
    execute "$LOGS/ssm_$name.log" "$PYTHON" scripts/train_ssm.py --data-dir "$DATA" --manifest "$MANIFEST" \
      --max-modes 40 --output "$ssm" "$@"
  fi
  if [[ ! -f ${prefix}_it4000_seed17.pt ]]; then
    log "train ${prefix}_it4000_seed17.pt"
    execute "$LOGS/unet_$name.log" "$PYTHON" scripts/train_unet.py --data-dir "$DATA" --manifest "$MANIFEST" \
      --lr 3e-4 --iterations 4000 --checkpoints 4000 --seed 17 --output-prefix "$prefix" "$@"
  fi
}

evaluate() {  # UNIT FOLD PRIOR LAYOUT PLANES RUNNER-ARGS...
  local unit=$1 fold=$2 prior=$3 layout=$4 planes=$5; shift 5
  local dir=$ROOT/$unit
  if [[ -f $dir/.complete ]]; then return 0; fi
  if [[ -e $dir ]]; then
    echo "partial output $dir: inspect it, move it aside, and rerun" >&2
    return 1
  fi
  local placement=(--plane-layout "$layout" --num-planes "$planes")
  if [[ $layout == axial ]]; then placement+=(--axial-fractions $(fractions "$planes")); fi
  log "evaluate $unit"
  execute "$LOGS/$unit.log" "$PYTHON" scripts/benchmark_sparse.py "${COMMON[@]}" --subset "fold$fold" \
    --run-dir "$dir" --ssm-model "$MODELS/ssm_$prior.pt" \
    --unet-models "$MODELS/unet_${prior}_lr3e-4_it4000_seed17.pt" "${placement[@]}" "$@"
  if [[ $DRY_RUN != 1 ]]; then touch "$dir/.complete"; fi
}

mkdir -p "$MODELS" "$LOGS"
if [[ $DRY_RUN != 1 ]] && command -v nvidia-smi > /dev/null 2>&1; then
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1)
  if (( free_mb < 7000 )); then
    echo "Refusing to start: only ${free_mb} MiB GPU memory is free; wait for other workloads to finish." >&2
    exit 2
  fi
fi
log "sensitivity v1 start (DRY_RUN=$DRY_RUN)"
"$PYTHON" scripts/make_sensitivity_subsets.py --manifest "$MANIFEST" --output-dir "$SUBSETS"

# Cross-fitted reference priors: evaluation fold k uses the other four development folds.
for fold in "${FOLDS[@]}"; do
  priors "reference_fold$fold" --train-folds $(other_folds "$fold")
done

# Study A: clean and perturbed sections on the protocol layouts.
for condition in clean random2 random4 inside2 outside2; do
  methods=("${ALL[@]}")
  if [[ $condition == clean ]]; then methods+=(mean_shape); fi
  for fold in "${FOLDS[@]}"; do
    for layout in orthogonal axial; do
      evaluate "A_${condition}_fold${fold}_$layout" "$fold" "reference_fold$fold" "$layout" 3 \
        --ssm-modes 20 --methods "${methods[@]}" ${CONDITION[$condition]}
    done
  done
done

# Study C: plane count; rbf needs non-coplanar constraints, so it is not run with one plane.
for fold in "${FOLDS[@]}"; do
  for spec in orthogonal:1 orthogonal:2 axial:1 axial:2 axial:4 axial:6; do
    layout=${spec%:*} planes=${spec#*:}
    methods=(inr interpolation)
    if (( planes > 1 )); then methods+=(rbf); fi
    methods+=(ssm unet gp prior_inr mean_shape)
    evaluate "C_${layout}${planes}_fold$fold" "$fold" "reference_fold$fold" "$layout" "$planes" \
      --ssm-modes 20 --methods "${methods[@]}"
  done
done

# Study B1: healthy-only versus size-matched mixed priors, on the pathology subjects.
priors healthy66 --train-subjects-file "$SUBSETS/healthy66.json"
for fold in "${FOLDS[@]}"; do
  priors "mixed66_fold$fold" --train-subjects-file "$SUBSETS/mixed66_fold$fold.json"
done
for fold in "${FOLDS[@]}"; do
  subjects=$(subjects_of "$SUBSETS/b1_eval_fold$fold.json")
  for prior in healthy66 "mixed66_fold$fold"; do
    for layout in orthogonal axial; do
      evaluate "B1_${prior%_fold*}_fold${fold}_$layout" "$fold" "$prior" "$layout" 3 \
        --ssm-modes 20 --methods "${PRIORS[@]}" --subjects $subjects
    done
  done
done

# Study B2: training-set size, with K = min(20, n - 1) shape modes.
for n in 10 25 50; do
  for fold in "${FOLDS[@]}"; do
    priors "size${n}_fold$fold" --train-subjects-file "$SUBSETS/size${n}_fold$fold.json"
  done
  for fold in "${FOLDS[@]}"; do
    for layout in orthogonal axial; do
      evaluate "B2_size${n}_fold${fold}_$layout" "$fold" "size${n}_fold$fold" "$layout" 3 \
        --ssm-modes $(( n - 1 < 20 ? n - 1 : 20 )) --methods "${PRIORS[@]}"
    done
  done
done
log "sensitivity v1 complete"
