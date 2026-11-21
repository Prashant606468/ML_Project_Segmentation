#!/usr/bin/env bash
# Reproduce every reported number and figure from the released checkpoints (no training).
#   bash scripts/reproduce_eval.sh <hf_repo_id>
# Requires data/ (scripts/download_data.sh). Takes ~30 min on one GPU.
set -e
cd "$(dirname "$0")/.."
REPO=${1:?usage: bash scripts/reproduce_eval.sh <hf_repo_id>}
mkdir -p logs

python3 scripts/checkpoints.py download --repo "$REPO"
python3 evaluate.py runs/*_s[0-9] > logs/eval.log 2>&1
for steps in 1 3 5 20; do
  python3 evaluate.py runs/yolo_diff_v8s_bnd_s[0-9] --steps "$steps" --tag "steps$steps" \
    --datasets CVC-ColonDB Kvasir CVC-ClinicDB >> logs/eval_sweep.log 2>&1
done
python3 evaluate.py runs/yolo_diff_v8s_bnd_s[0-9] --n_samples 5 --tag ens5 \
  --datasets CVC-ColonDB Kvasir CVC-ClinicDB >> logs/eval_sweep.log 2>&1

python3 analysis/summarize.py > logs/summary.log 2>&1
python3 analysis/dataset_stats.py >> logs/summary.log 2>&1
python3 analysis/error_analysis.py >> logs/summary.log 2>&1
echo "Done: tables and figures in results/"
