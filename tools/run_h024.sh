#!/usr/bin/env bash
# H-024, laptop side: A100 instructions whose K is two groups of the unit, split into one vector
# file per pass (tools/kv260/board_cases_x.py chain), each pass through rtl/tm_pl_x.v in
# Verilator (built by tools/run_pl_x.sh as the instruction with K = L), and the board's case
# files.
#   tools/run_h024.sh <A100 data dir> [model cases]
# Results in results/rtl/tm_pl_x-h024-<commit>.jsonl; vectors and case files in build/h024/.
set -u
cd "$(dirname "$0")/.."
g=$1 n=${2:-20000}
out=results/rtl/tm_pl_x-h024-$(git rev-parse --short HEAD).jsonl
: > "$out"
mkdir -p build/h024
for spec in "f32.f16.f16.f32|m16n8k8|m16n8k16" "f32.bf16.bf16.f32|m16n8k8|m16n8k16" \
            "f16.f16.f16.f16|m16n8k8|m16n8k16" "f32.tf32.tf32.f32|m16n8k4|m16n8k8"; do
    IFS="|" read -r types unit inst <<< "$spec"
    tag=Ampere-$types-$unit
    bash tools/run_pl_x.sh "$out" "$n" "Ampere|$types|$unit" || exit 2
    python tools/kv260/board_cases_x.py chain Ampere "$g/$inst.$types" "build/h024/$inst.$types" || exit 2
    read -r L aw bw < <(python tools/rtl_params.py Ampere "$types" "$unit" | tail -1)
    for pass in 1 2; do
        v=build/h024/$inst.$types-$pass
        r=$(build/pl-x/$tag-tm_pl_x/tb "$v.txt" 5 70 70 2> "$v.err")
        echo "{\"config\": \"$inst.$types\", \"pass\": $pass, \"unit\": \"$tag\", \"exit\": $?, \"result\": $r}" \
            | tee -a "$out"
        python tools/kv260/board_cases_x.py pack "$v.txt" "$L" "$aw" "$bw" 0 "$v.bin" || exit 2
    done
done
sha256sum build/h024/*.bin
