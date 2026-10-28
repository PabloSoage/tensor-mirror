#!/usr/bin/env bash
# H-028: a loop of S block-scale mma.sync per output tile (gpu/mma_block_scale.cu, last
# argument) against the model chained the same way, for every instruction of the build; and
# S = 1 against the kernel before the loop (git <base>), byte for byte. On the GPU host, in
# the venv:
#   CUDA_VISIBLE_DEVICES=<gpu> tools/run_h028.sh <root> [output tiles] [steps] [base]
# Summary in <root>/summary.jsonl.
set -u
cd "$(dirname "$0")/.."
root=$1 tiles=${2:-512} steps=${3:-8} base=${4:-}
mkdir -p "$root/bin"
gen="-gencode arch=compute_120a,code=sm_120a"
nvcc -O2 $gen -o "$root/bin/mma_block_scale" gpu/mma_block_scale.cu || exit 2
: > "$root/summary.jsonl"
run() {  # binary, dir, output tiles, [steps]
    "$1" "$inst" "$3" "$2/a.bin" "$2/b.bin" "$2/c.bin" "$2/sa.bin" "$2/sb.bin" "$2/sel.bin" "$2/d${5:-}.bin" ${4:-}
}
for inst in $("$root/bin/mma_block_scale" --list); do
    d=$root/s$steps/$inst
    python tools/bs_cases.py gen --inst "$inst" --tiles $((tiles * steps)) --seed 1 --out "$d" > /dev/null || exit 2
    nt=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['tiles'] // $steps)" "$d/cases.json")
    run "$root/bin/mma_block_scale" "$d" "$nt" "$steps" > "$d/run.json" || exit 2
    python tools/bs_cases.py check --dir "$d" --steps "$steps" --jobs 24 >> "$root/summary.jsonl"
done
if [ -n "$base" ]; then  # S = 1 equals the kernel before the loop
    git show "$base:gpu/mma_block_scale.cu" > "$root/bin/mma_block_scale_base.cu"
    nvcc -O2 $gen -o "$root/bin/mma_block_scale_base" "$root/bin/mma_block_scale_base.cu" || exit 2
    for inst in $("$root/bin/mma_block_scale" --list); do
        d=$root/s1/$inst
        python tools/bs_cases.py gen --inst "$inst" --tiles "$tiles" --seed 1 --out "$d" > /dev/null
        nt=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['tiles'])" "$d/cases.json")
        run "$root/bin/mma_block_scale" "$d" "$nt" > /dev/null
        run "$root/bin/mma_block_scale_base" "$d" "$nt" "" _base > /dev/null
        same=$(cmp -s "$d/d.bin" "$d/d_base.bin" && echo true || echo false)
        echo "{\"inst\": \"$inst\", \"steps\": 1, \"identical_to_base\": $same}" >> "$root/summary.jsonl"
    done
fi
nvidia-smi --query-gpu=name,driver_version,vbios_version --format=csv > "$root/gpu.csv"
nvcc --version | tail -1 > "$root/nvcc.txt"
echo "done $root"
