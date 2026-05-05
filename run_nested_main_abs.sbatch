#!/bin/bash
#SBATCH --job-name=obj-traj-abs
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --output=/data2/dohyeon/object_traj/logs/%x_%j.out
#SBATCH --error=/data2/dohyeon/object_traj/logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="/data2/dohyeon/object_traj"
cd "$PROJECT_ROOT"
mkdir -p logs

# Usage:
#   sbatch run_nested_main_abs.sbatch data/20200820_144100
#   sbatch run_nested_main_abs.sbatch data/20201015_150611 --angle 50 --eef-dir mz
#
# The first argument can be either:
#   1. a normal dataset directory containing object_pose/poses.npz and camera.json
#   2. an outer directory containing multiple dataset directories, e.g.
#      data/20200820_144100/{foundpose,freepose,gt,megapose,ours,unihope}
#
# Any remaining arguments are forwarded to simulation/main_abs_date.py so all inner
# datasets run with the same simulation conditions.

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch $0 <data_dir_or_outer_dir> [main_abs.py args...]"
    exit 2
fi

DATA_ROOT="$1"
shift

if [[ "$DATA_ROOT" != /* ]]; then
    DATA_ROOT="$PROJECT_ROOT/$DATA_ROOT"
fi

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "Data directory does not exist: $DATA_ROOT"
    exit 1
fi

if [[ -d ".venv" ]]; then
    source .venv/bin/activate
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate object-traj
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"

echo "START JOB ${SLURM_JOB_ID:-local}"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "DATA_ROOT=$DATA_ROOT"
echo "EXTRA_ARGS=$*"
nvidia-smi || true

declare -a DATASETS=()

if [[ -f "$DATA_ROOT/object_pose/poses.npz" && -f "$DATA_ROOT/camera.json" ]]; then
    DATASETS+=("$DATA_ROOT")
else
    for dir in "$DATA_ROOT"/*; do
        [[ -d "$dir" ]] || continue
        if [[ -f "$dir/object_pose/poses.npz" && -f "$dir/camera.json" ]]; then
            DATASETS+=("$dir")
        fi
    done
fi

if [[ ${#DATASETS[@]} -eq 0 ]]; then
    echo "No runnable dataset directories found under: $DATA_ROOT"
    exit 1
fi

OUTER_NAME="$(basename "$DATA_ROOT")"
VIDEO_DIR="$PROJECT_ROOT/videos/$OUTER_NAME"

echo "Found ${#DATASETS[@]} dataset(s):"
printf '  %s\n' "${DATASETS[@]}"

for dataset in "${DATASETS[@]}"; do
    name="${OUTER_NAME}_$(basename "$dataset")"
    run_dir="$VIDEO_DIR/$name"
    log_dir="$run_dir/logs"
    mkdir -p "$log_dir"

    echo
    echo "==== Running $name ===="
    python simulation/main_abs_date.py "$dataset" \
        --name "$name" \
        --video-dir "$VIDEO_DIR" \
        "$@" 2>&1 | tee "$log_dir/main_abs.log"
done

echo
echo "DONE"
