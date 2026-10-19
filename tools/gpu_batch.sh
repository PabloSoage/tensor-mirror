#!/usr/bin/env bash
# Generate, run and check a set of mma.sync instructions.
#   tools/gpu_batch.sh gen   <arch> <root> <tiles> <inst>...   cases in <root>/<inst>/
#   tools/gpu_batch.sh run   <root> <mma_sync binary> [<mma_sm70 binary>]   on the GPU host
#   tools/gpu_batch.sh check <root> [jobs]                     summary in <root>/summary.jsonl
# The run step needs only the binaries and the case files, so it can be done on another
# machine. m8n8k4 instructions go to the mma_sm70 harness, the rest to mma_sync.
set -eu
cmd=$1
shift
here=$(cd "$(dirname "$0")" && pwd)
case $cmd in
gen)
    arch=$1 root=$2 tiles=$3
    shift 3
    for inst in "$@"; do
        python "$here/gpu_cases.py" gen --arch "$arch" --inst "$inst" --tiles "$tiles" --seed ${SEED:-1} \
            --out "$root/$inst" > /dev/null
    done
    ;;
run)
    root=$1 sync=$2 sm70=${3:-}
    for dir in "$root"/*/; do
        inst=$(basename "$dir")
        tiles=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['tiles'])" "$dir/cases.json")
        if [[ $inst == m8n8k4.* ]]; then bin=$sm70 name=${inst#m8n8k4.}; else bin=$sync name=$inst; fi
        "$bin" "$name" "$tiles" "$dir/a.bin" "$dir/b.bin" "$dir/c.bin" "$dir/d.bin" > "$dir/run.json"
    done
    nvidia-smi --query-gpu=name,driver_version,vbios_version --format=csv > "$root/gpu.csv"
    ;;
check)
    root=$1 jobs=${2:-8}
    : > "$root/summary.jsonl"
    for dir in "$root"/*/; do
        python "$here/gpu_cases.py" check --dir "$dir" --mmasim --jobs "$jobs" >> "$root/summary.jsonl" || true
    done
    cat "$root/summary.jsonl"
    ;;
esac
