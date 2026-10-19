#!/usr/bin/env bash
# On the GPU host: build the Volta harness and run it on case directories made by
# `gpu_cases.py gen` (one per variant: <root>/<variant>/{a,b,c}.bin, cases.json).
#   tools/run_v100.sh <root> [gpu index]
set -eu
root=$1
gpu=${2:-0}
here=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$here/build"
nvcc --version | tail -2
nvcc -O2 -arch=sm_70 -o "$here/build/mma_sm70" "$here/gpu/mma_sm70.cu"
for dir in "$root"/*/; do
    variant=$(basename "$dir")
    tiles=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['tiles'])" "$dir/cases.json")
    CUDA_VISIBLE_DEVICES=$gpu "$here/build/mma_sm70" "$variant" "$tiles" \
        "$dir/a.bin" "$dir/b.bin" "$dir/c.bin" "$dir/d.bin" | tee "$dir/run.json"
done
nvidia-smi --query-gpu=index,name,driver_version,vbios_version --format=csv -i "$gpu" | tee "$root/gpu.csv"
