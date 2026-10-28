#!/usr/bin/env bash
# H-029, the GPU side: GEMM jobs of tools/gemm_cases.py cut into 16 x 8 tiles, each through
# a loop of mma.sync over the whole of K (gpu/mma_sync.cu with steps), D gathered back and
# every tile checked against the model (tools/gpu_cases.py check). Run on the GPU host, in
# the venv:
#   CUDA_VISIBLE_DEVICES=<gpu> tools/run_h029_gpu.sh <arch> <sm> <k> <root> [jobs]
#   e.g. tools/run_h029_gpu.sh Ampere 80 16 build/h029/a100 96
# The jobs are those of seed 1, the same on every host. Summary in <root>/summary.jsonl.
set -u
cd "$(dirname "$0")/.."
arch=$1 sm=$2 k=$3 root=$4 jobs=${5:-96}
mkdir -p "$root/bin"
nvcc -O2 -gencode arch=compute_$sm,code=sm_$sm -DTM_SM="${sm%a}" -o "$root/bin/mma_sync" gpu/mma_sync.cu || exit 2
: > "$root/summary.jsonl"
for types in f16.f16.f16.f16 f32.f16.f16.f32; do
    d=$root/$types inst=m16n8k$k.$types
    python tools/gemm_cases.py gen --arch "$arch" --types "$types" --jobs "$jobs" --seed 1 --out "$d" > /dev/null || exit 2
    read -r nt steps < <(python tools/gemm_cases.py tiles --dir "$d" --inst "$inst" \
        | python -c "import json,sys; j=json.load(sys.stdin); print(j['tiles'], j['steps'])")
    t=$d/$inst
    "$root/bin/mma_sync" "$inst" "$nt" "$t/a.bin" "$t/b.bin" "$t/c.bin" "$t/d.bin" "$steps" > "$t/run.json" || exit 2
    python tools/gemm_cases.py gather --dir "$d" --inst "$inst" || exit 2
    python tools/gpu_cases.py check --dir "$t" --jobs 16 >> "$root/summary.jsonl"
    sha256sum "$d/gemm_a.bin" "$d/gemm_b.bin" "$d/gemm_c.bin" "$t/gemm_d.bin" | sed "s|$root/||" >> "$root/sha256.txt"
done
nvidia-smi --query-gpu=name,driver_version,vbios_version --format=csv -i "${CUDA_VISIBLE_DEVICES:-0}" \
    > "$root/gpu.csv" 2>/dev/null || nvidia-smi --query-gpu=name,driver_version,vbios_version --format=csv > "$root/gpu.csv"
nvcc --version | tail -1 > "$root/nvcc.txt"
echo "done $root"
