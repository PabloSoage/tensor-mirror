#!/usr/bin/env bash
# H-027: a loop of S mma.sync per tile (gpu/mma_sync.cu, last argument) against the model
# chained over the whole row, for every instruction of one build; and S = 1 against the
# kernel before the loop (git <base>), byte for byte. Run on the GPU host, in the venv:
#   CUDA_VISIBLE_DEVICES=<gpu> tools/run_h027.sh <arch> <sm> <root> [tiles] [steps] [base]
#   e.g. tools/run_h027.sh "RTX Blackwell" 120a build/h027/rtx5070ti 1024 16 d647236
# Summary in <root>/summary.jsonl.
set -u
cd "$(dirname "$0")/.."
arch=$1 sm=$2 root=$3 tiles=${4:-1024} steps=${5:-16} base=${6:-}
mkdir -p "$root/bin"
gen="-gencode arch=compute_$sm,code=sm_$sm"
nvcc -O2 $gen -DTM_SM="${sm%a}" -o "$root/bin/mma_sync" gpu/mma_sync.cu || exit 2
: > "$root/summary.jsonl"
for inst in $("$root/bin/mma_sync" --list); do
    d=$root/s$steps/$inst
    python tools/gpu_cases.py gen --arch "$arch" --inst "$inst" --tiles "$tiles" --seed 1 --steps "$steps" \
        --out "$d" > /dev/null || exit 2
    nt=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['tiles'])" "$d/cases.json")  # with the directed
    "$root/bin/mma_sync" "$inst" "$nt" "$d/a.bin" "$d/b.bin" "$d/c.bin" "$d/d.bin" "$steps" > "$d/run.json" || exit 2
    python tools/gpu_cases.py check --dir "$d" --jobs 16 >> "$root/summary.jsonl"
done
if [ -n "$base" ]; then  # prediction 2: S = 1 equals the kernel before the loop
    git show "$base:gpu/mma_sync.cu" > "$root/bin/mma_sync_base.cu"
    nvcc -O2 $gen -DTM_SM="${sm%a}" -o "$root/bin/mma_sync_base" "$root/bin/mma_sync_base.cu" || exit 2
    for inst in $("$root/bin/mma_sync" --list); do
        d=$root/s1/$inst
        python tools/gpu_cases.py gen --arch "$arch" --inst "$inst" --tiles "$tiles" --seed 1 --out "$d" > /dev/null
        nt=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['tiles'])" "$d/cases.json")
        "$root/bin/mma_sync" "$inst" "$nt" "$d/a.bin" "$d/b.bin" "$d/c.bin" "$d/d.bin" > /dev/null
        "$root/bin/mma_sync_base" "$inst" "$nt" "$d/a.bin" "$d/b.bin" "$d/c.bin" "$d/d_base.bin" > /dev/null
        same=$(cmp -s "$d/d.bin" "$d/d_base.bin" && echo true || echo false)
        echo "{\"inst\": \"$inst\", \"steps\": 1, \"identical_to_base\": $same}" >> "$root/summary.jsonl"
    done
fi
nvidia-smi --query-gpu=name,driver_version,vbios_version --format=csv -i "${CUDA_VISIBLE_DEVICES:-0}" \
    > "$root/gpu.csv" 2>/dev/null || nvidia-smi --query-gpu=name,driver_version,vbios_version --format=csv > "$root/gpu.csv"
nvcc --version | tail -1 > "$root/nvcc.txt"
echo "done $root"
