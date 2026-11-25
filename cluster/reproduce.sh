#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
group=${1:?expected grayscale or cifar10}
run=${2:?expected HTCondor cluster ID}
case "$group" in
    grayscale) datasets=(mnist fashion_mnist) ;;
    cifar10) datasets=(cifar10) ;;
    *) printf 'Unknown dataset group: %s\n' "$group" >&2; exit 2 ;;
esac

python=.venv/bin/python
export JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1 BENDFLOW_PLOT_TEX=0
"$python" -c 'import jax; print(jax.__version__, jax.devices())'

root="reproduction/$run"
checkpoints="$root/checkpoints"
results="$root/results"
mkdir -p "$checkpoints" "$results"

for dataset in "${datasets[@]}"; do
    if [[ -f "$root/$dataset.complete" ]]; then
        continue
    fi
    for model in unet dit; do
        "$python" experiments/train.py --dataset "$dataset" --model "$model" --out "$checkpoints"
    done
    div_batch=8
    if [[ "$dataset" == cifar10 ]]; then
        # Match the archived CIFAR-10 run's smaller divergence-check batch.
        div_batch=1
    fi
    "$python" experiments/exact_field.py --dataset "$dataset" --out "$results/exact_field_$dataset" --div-batch "$div_batch"
    "$python" experiments/trained_networks.py --dataset "$dataset" --checkpoints "$checkpoints" --out "$results/trained_networks_$dataset"
    "$python" experiments/sampler_comparison.py --dataset "$dataset" --checkpoints "$checkpoints" --out "$results/sampler_comparison_$dataset"
    "$python" experiments/schedule_baseline.py --dataset "$dataset" --checkpoints "$checkpoints" --out "$results/schedule_baseline_$dataset"
    settings="$results/schedule_baseline_${dataset}_n128"
    "$python" experiments/schedule_baseline.py --dataset "$dataset" --checkpoints "$checkpoints" --out "$settings" \
        --nfe 5 10 20 40 --num-samples 128 --seed 4 --h-max-grid 0.25 0.5
    "$python" experiments/sample_grids.py --dataset "$dataset" --checkpoints "$checkpoints" --out "$results/sample_grids_$dataset" --settings "$settings"
    touch "$root/$dataset.complete"
done
