#!/usr/bin/env bash
# Full experiment: 6 configurations x 3 seeds, one queue per seed run in parallel,
# followed by evaluation and analysis.
#   bash run_all.sh            (logs in logs/)
set -e
cd "$(dirname "$0")"
mkdir -p logs

CONFIGS=(
  "--model unet"
  "--model pranet"
  "--model yolo_seg"
  "--model yolo_diff --no_boundary"
  "--model yolo_diff"
  "--model yolo_diff --aug strong"
)

queue() {
  local seed=$1
  for cfg in "${CONFIGS[@]}"; do
    python3 train.py $cfg --seed "$seed"
  done
}

for seed in 0 1 2; do
  queue "$seed" > "logs/train_seed$seed.log" 2>&1 &
done
wait

python3 evaluate.py runs/*_s[0-9] > logs/eval.log 2>&1
# DDIM step / ensemble sweep for the proposed model
for steps in 1 3 5 20; do
  python3 evaluate.py runs/yolo_diff_v8s_bnd_s[0-9] --steps "$steps" --tag "steps$steps" \
    --datasets CVC-ColonDB Kvasir CVC-ClinicDB >> logs/eval_sweep.log 2>&1
done
python3 evaluate.py runs/yolo_diff_v8s_bnd_s[0-9] --n_samples 5 --tag ens5 \
  --datasets CVC-ColonDB Kvasir CVC-ClinicDB >> logs/eval_sweep.log 2>&1

python3 analysis/summarize.py > logs/summary.log 2>&1
python3 analysis/dataset_stats.py >> logs/summary.log 2>&1
python3 analysis/error_analysis.py >> logs/summary.log 2>&1
