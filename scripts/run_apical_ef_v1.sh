#!/usr/bin/env bash
# Apical-view EF study v1 (benchmarks/mitea_apical_ef_v1.md): development folds 1-4 only; the
# held-out test set is never selected. Run from the repository root, for example in tmux:
#   tmux new-session -d -s apical_ef_v1 \
#     'scripts/run_apical_ef_v1.sh 2>&1 | tee -a runs/mitea_apical_ef_v1/run.log'
# Steps run one at a time. Finished units are skipped, so rerunning resumes; a partial unit stops
# the run for inspection. DRY_RUN=1 prints the commands without running them.
set -euo pipefail

PYTHON=.venv/bin/python
DATA=dataset/mitea
MANIFEST=benchmarks/splits/mitea_cohort_v1.json
ROOT=runs/mitea_apical_ef_v1
MODELS=$ROOT/models
LOGS=$ROOT/logs
DRY_RUN=${DRY_RUN:-0}

log() { echo "[$(date '+%F %T')] $*"; }

execute() {  # LOGFILE COMMAND...
  local output=$1; shift
  if [[ $DRY_RUN == 1 ]]; then echo "  $* > $output"; return 0; fi
  "$@" > "$output" 2>&1
}

other_folds() {
  local fold
  for fold in 0 1 2 3 4; do if [[ $fold != "$1" ]]; then printf '%s ' "$fold"; fi; done
}

mkdir -p "$MODELS" "$LOGS"
if [[ $DRY_RUN != 1 ]] && command -v nvidia-smi > /dev/null 2>&1; then
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1)
  if (( free_mb < 7000 )); then
    echo "Refusing to start: only ${free_mb} MiB GPU memory is free; wait for other workloads to finish." >&2
    exit 2
  fi
fi
log "apical EF v1 start (DRY_RUN=$DRY_RUN)"

for fold in 1 2 3 4; do
  train=$(other_folds "$fold")
  ssm=$MODELS/ssm_cavity_fold$fold.pt prefix=$MODELS/unet_cavity_fold${fold}_lr3e-4
  if [[ ! -f $ssm ]]; then
    log "build $ssm"
    execute "$LOGS/ssm_cavity_fold$fold.log" "$PYTHON" scripts/train_ssm.py --data-dir "$DATA" \
      --manifest "$MANIFEST" --train-folds $train --target cavity --phases ED ES --max-modes 40 --output "$ssm"
  fi
  if [[ ! -f ${prefix}_it4000_seed17.pt ]]; then
    log "train ${prefix}_it4000_seed17.pt"
    execute "$LOGS/unet_cavity_fold$fold.log" "$PYTHON" scripts/train_unet.py --data-dir "$DATA" \
      --manifest "$MANIFEST" --train-folds $train --target cavity --phases ED ES --observation apical \
      --lr 3e-4 --iterations 4000 --checkpoints 4000 --seed 17 --output-prefix "$prefix"
  fi
done

for fold in 1 2 3 4; do
  dir=$ROOT/E_fold$fold
  if [[ -f $dir/.complete ]]; then continue; fi
  if [[ -e $dir ]]; then
    echo "partial output $dir: inspect it, move it aside, and rerun" >&2
    exit 1
  fi
  log "evaluate E_fold$fold"
  execute "$LOGS/E_fold$fold.log" "$PYTHON" scripts/benchmark_apical.py --data-dir "$DATA" --manifest "$MANIFEST" \
    --subset "fold$fold" --run-dir "$dir" --ssm-model "$MODELS/ssm_cavity_fold$fold.pt" \
    --unet-model "$MODELS/unet_cavity_fold${fold}_lr3e-4_it4000_seed17.pt" --seed 17 --device cuda
  if [[ $DRY_RUN != 1 ]]; then touch "$dir/.complete"; fi
done
if [[ $DRY_RUN != 1 ]]; then "$PYTHON" scripts/summarize_apical_ef.py --root "$ROOT"; fi
log "apical EF v1 complete"
