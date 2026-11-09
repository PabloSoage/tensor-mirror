#!/usr/bin/env bash
# H-037, the GPU side: block-scale GEMM jobs of tools/bs_gemm.py cut into chained 16 x 8
# tiles, run through gpu/mma_block_scale.cu with steps on an sm_120a GPU, D gathered back,
# and every output tile checked against the chained model (tools/bs_cases.py check --steps).
# Run on the GPU host, in the venv:
#   CUDA_VISIBLE_DEVICES=<gpu> tools/run_bs_gemm_gpu.sh <root> <jobs> <inst>...
# Summary in <root>/summary.jsonl; one directory per instruction.
set -u
cd "$(dirname "$0")/.."
root=$1 jobs=$2
shift 2
mkdir -p "$root/bin"
nvcc -O2 -gencode arch=compute_120a,code=sm_120a -o "$root/bin/mma_block_scale" gpu/mma_block_scale.cu || exit 2
: > "$root/summary.jsonl"
: > "$root/sha256.txt"
for inst in "$@"; do
    d=$root/$inst t=$root/$inst/tiles
    python tools/bs_gemm.py gen --inst "$inst" --jobs "$jobs" --out "$d" > /dev/null || exit 2
    python tools/bs_gemm.py tiles --dir "$d" > /dev/null || exit 2
    read -r ni steps < <(python -c "import json,sys; m=json.load(open(sys.argv[1])); print(m['tiles'], m['steps'])" "$t/cases.json")
    "$root/bin/mma_block_scale" "$inst" $((ni / steps)) "$t/a.bin" "$t/b.bin" "$t/c.bin" "$t/sa.bin" "$t/sb.bin" \
        "$t/sel.bin" "$t/d.bin" "$steps" > "$t/run.json" || exit 2
    python tools/bs_gemm.py gather --dir "$d" || exit 2
    python tools/bs_cases.py check --dir "$t" --steps "$steps" --jobs 16 >> "$root/summary.jsonl"
    sha256sum "$d"/gemm_*.bin "$t/gemm_d.bin" | sed "s|$root/||" >> "$root/sha256.txt"
done
nvidia-smi --query-gpu=name,driver_version,vbios_version --format=csv -i "${CUDA_VISIBLE_DEVICES:-0}" > "$root/gpu.csv"
nvcc --version | tail -1 > "$root/nvcc.txt"
echo "done $root"
