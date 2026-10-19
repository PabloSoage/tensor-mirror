#!/usr/bin/env bash
# Build rtl/tfdpa_f16_f32.sv with Verilator and check it against the integer model, then
# check that the testbench catches mutants (parameter and code mutants).
#   tools/run_rtl.sh [cases]
set -u
cd "$(dirname "$0")/.."
n=${1:-200000}
b=build/rtl
mkdir -p $b
out=results/rtl/tfdpa_f16_f32-$(git rev-parse --short HEAD).jsonl
mkdir -p results/rtl
: > "$out"

build() {  # name, extra verilator flags...
    local name=$1
    shift
    verilator --cc --exe --build -O3 -Wall -Wno-UNUSEDSIGNAL --top-module tfdpa_f16_f32 \
        -Mdir "$b/$name" -o tb "$@" "$PWD/rtl/tfdpa_f16_f32.sv" "$PWD/tb/tb_tfdpa_f16_f32.cpp" > "$b/$name.log" 2>&1 \
        || { cat "$b/$name.log"; exit 2; }
}

python tools/rtl_vectors.py --cases "$n" --seed 1 --out $b/vectors.txt
build ref
r=$("$b/ref/tb" $b/vectors.txt 2>"$b/ref.err")
echo "{\"dut\": \"ref\", \"result\": $r}" | tee -a "$out"

# Mutants: each must give mismatches (except e_zero with C in f32, unobservable on Volta)
for m in "F22:-GF=22" "F24:-GF=24" "EZERO130:-GEZERO=-130" "ROUND_ALIGN:+define+MUT_ROUND_ALIGN"; do
    name=${m%%:*}
    build "$name" "${m#*:}"
    r=$("$b/$name/tb" $b/vectors.txt 0 2>/dev/null)
    echo "{\"dut\": \"$name\", \"result\": $r}" | tee -a "$out"
done
