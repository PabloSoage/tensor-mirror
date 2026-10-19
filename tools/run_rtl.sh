#!/usr/bin/env bash
# Build rtl/tfdpa_f16_f32.sv with Verilator, combinational and pipelined, and check both
# against the integer model; then check that the testbench catches mutants of the
# pipelined unit (parameter and code mutants).
#   tools/run_rtl.sh [cases]
set -u
cd "$(dirname "$0")/.."
n=${1:-200000}
b=build/rtl
mkdir -p $b results/rtl
out=results/rtl/tfdpa_f16_f32-$(git rev-parse --short HEAD).jsonl
: > "$out"

build() {  # name, latency, extra verilator flags...
    local name=$1 lat=$2
    shift 2
    verilator --cc --exe --build -O3 -Wall -Wno-UNUSEDSIGNAL --top-module tfdpa_f16_f32 \
        -Mdir "$b/$name" -o tb -CFLAGS "-DTB_LAT=$lat" "$@" \
        "$PWD/rtl/tfdpa_f16_f32.sv" "$PWD/tb/tb_tfdpa_f16_f32.cpp" > "$b/$name.log" 2>&1 \
        || { cat "$b/$name.log"; exit 2; }
}
run() {  # name, latency, extra flags...
    local name=$1
    build "$@"
    r=$("$b/$name/tb" $b/vectors.txt 3 2>"$b/$name.err")
    echo "{\"dut\": \"$name\", \"result\": $r}" | tee -a "$out"
}

python tools/rtl_vectors.py --cases "$n" --seed 1 --out $b/vectors.txt
run comb 0 -GPIPE=0
run pipe 7 -GPIPE=1
# Mutants: each must give mismatches (except e_zero with C in f32, unobservable on Volta)
run pipe_F22 7 -GPIPE=1 -GF=22
run pipe_F24 7 -GPIPE=1 -GF=24
run pipe_EZERO130 7 -GPIPE=1 -GEZERO=-130
run pipe_ROUND_ALIGN 7 -GPIPE=1 +define+MUT_ROUND_ALIGN
run pipe_LAT6 6 -GPIPE=1  # the testbench must also notice a wrong latency
