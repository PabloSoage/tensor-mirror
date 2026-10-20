#!/usr/bin/env bash
# Generate, run and check the block-scale instructions (gpu/mma_block_scale.cu).
#   tools/bs_batch.sh gen   <root> <tiles> <inst>...     cases in <root>/<inst>/ (SEED, default 1)
#   tools/bs_batch.sh probe <root> <inst>...             scale-layout probes in <root>/<inst>/
#   tools/bs_batch.sh run   <root> <binary>              on the GPU host
#   tools/bs_batch.sh check <root> [jobs]                summary in <root>/summary.jsonl
# Probe directories are recognised by their cases.json and checked with probe-check.
set -eu
cmd=$1
shift
here=$(cd "$(dirname "$0")" && pwd)
case $cmd in
gen)
    root=$1 tiles=$2
    shift 2
    for inst in "$@"; do
        python "$here/bs_cases.py" gen --inst "$inst" --tiles "$tiles" --seed ${SEED:-1} --out "$root/$inst" > /dev/null
    done
    ;;
probe)
    root=$1
    shift
    for inst in "$@"; do python "$here/bs_cases.py" probe-gen --inst "$inst" --out "$root/$inst" > /dev/null; done
    ;;
run)
    root=$1 bin=$2
    for dir in "$root"/*/; do
        inst=$(basename "$dir")
        tiles=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['tiles'])" "$dir/cases.json")
        "$bin" "$inst" "$tiles" "$dir"/a.bin "$dir"/b.bin "$dir"/c.bin "$dir"/sa.bin "$dir"/sb.bin "$dir"/sel.bin \
            "$dir"/d.bin > "$dir/run.json"
    done
    nvidia-smi --query-gpu=name,driver_version,vbios_version --format=csv > "$root/gpu.csv"
    ;;
check)
    root=$1 jobs=${2:-8}
    : > "$root/summary.jsonl"
    for dir in "$root"/*/; do
        if grep -q '"probe"' "$dir/cases.json"; then
            python "$here/bs_cases.py" probe-check --dir "$dir" >> "$root/summary.jsonl" || true
        else
            python "$here/bs_cases.py" check --dir "$dir" --mmasim --jobs "$jobs" >> "$root/summary.jsonl" || true
        fi
    done
    cat "$root/summary.jsonl"
    ;;
esac
