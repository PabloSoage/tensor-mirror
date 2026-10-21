#!/usr/bin/env bash
# H-020, laptop side: rtl/tm_pl_x.v in Verilator (tools/run_pl_x.sh) as each instruction whose
# K is one group of the unit, then against what its GPU gave, and the board's case files.
#   tools/run_h020.sh <GPU data dir: holds a100/ and rtx5070ti/> [model cases]
# Results in results/rtl/tm_pl_x-h020-<commit>.jsonl; vectors and case files in build/h020/.
set -u
cd "$(dirname "$0")/.."
g=$1 n=${2:-20000}
out=results/rtl/tm_pl_x-h020-$(git rev-parse --short HEAD).jsonl
: > "$out"
mkdir -p build/h020
for spec in "Ampere|f32.f16.f16.f32|m16n8k8|a100" "Ampere|f32.bf16.bf16.f32|m16n8k8|a100" \
            "Ampere|f16.f16.f16.f16|m16n8k8|a100" "Ampere|f32.tf32.tf32.f32|m16n8k4|a100" \
            "RTX Blackwell|f32.e4m3.e4m3.f32|m16n8k32|rtx5070ti"; do
    IFS="|" read -r arch types shape gpu <<< "$spec"
    tag=$(echo "$arch-$types-$shape" | tr ' ' '_')
    bash tools/run_pl_x.sh "$out" "$n" "$arch|$types|$shape" || exit 2
    python tools/kv260/board_cases_x.py plain "$g/$gpu/$shape.$types" "build/h020/$tag.txt" || exit 2
    r=$(build/pl-x/$tag-tm_pl_x/tb "build/h020/$tag.txt" 5 70 70 2> "build/h020/$tag.err")
    echo "{\"config\": \"$tag\", \"dut\": \"tm_pl_x\", \"vectors\": \"$gpu\", \"exit\": $?, \"result\": $r}" \
        | tee -a "$out"
    read -r L aw bw < <(python tools/rtl_params.py "$arch" "$types" "$shape" | tail -1)
    python tools/kv260/board_cases_x.py pack "build/h020/$tag.txt" "$L" "$aw" "$bw" 0 "build/h020/$tag.bin" \
        || exit 2
done
sha256sum build/h020/*.bin
