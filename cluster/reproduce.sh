#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
group=${1:?expected grayscale or cifar10}
run=${2:?expected HTCondor cluster ID}
schedule=${3:-linear}
case "$schedule" in
    linear|cosine|ve) ;;
    *) printf 'Unknown schedule: %s\n' "$schedule" >&2; exit 2 ;;
esac
case "$group" in
    grayscale) datasets=(mnist fashion_mnist) ;;
    cifar10) datasets=(cifar10) ;;
    *) printf 'Unknown dataset group: %s\n' "$group" >&2; exit 2 ;;
esac

python=.venv/bin/python
export PYTHONUNBUFFERED=1 BENDFLOW_PLOT_TEX=0
"$python" -c 'import torch; torch.cuda.init(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'

experiment() {
    local script=$1
    shift
    "$python" "experiments/$script.py" --device cuda --schedule "$schedule" "$@"
}

root="reproduction/pytorch/$schedule/$run"
checkpoints="$root/checkpoints"
results="$root/results"
mkdir -p "$checkpoints" "$results"

for dataset in "${datasets[@]}"; do
    if [[ -f "$root/$dataset.complete" ]]; then
        continue
    fi
    for model in unet dit; do
        experiment train --dataset "$dataset" --model "$model" --out "$checkpoints"
    done
    div_batch=8
    if [[ "$dataset" == cifar10 ]]; then
        # Match the archived CIFAR-10 run's smaller divergence-check batch.
        div_batch=1
    fi
    experiment exact_field --dataset "$dataset" --out "$results/exact_field_$dataset" --div-batch "$div_batch"
    experiment trained_networks --dataset "$dataset" --checkpoints "$checkpoints" --out "$results/trained_networks_$dataset"
    experiment sampler_comparison --dataset "$dataset" --checkpoints "$checkpoints" --out "$results/sampler_comparison_$dataset"
    experiment schedule_baseline --dataset "$dataset" --checkpoints "$checkpoints" --out "$results/schedule_baseline_$dataset"
    settings="$results/schedule_baseline_${dataset}_n128"
    experiment schedule_baseline --dataset "$dataset" --checkpoints "$checkpoints" --out "$settings" \
        --nfe 5 10 20 40 --num-samples 128 --seed 4 --h-max-grid 0.25 0.5
    experiment sample_grids --dataset "$dataset" --checkpoints "$checkpoints" --out "$results/sample_grids_$dataset" --settings "$settings"
    touch "$root/$dataset.complete"
done
