#!/usr/bin/env bash
# Build rtl/tfdpa_f16.sv with Verilator for Volta's three m8n8k4 variants, combinational
# and pipelined, and check them against the integer model; then check that the testbench
# catches mutants of the pipelined unit.
#   tools/run_rtl.sh [cases]
set -u
cd "$(dirname "$0")/.."
n=${1:-200000}
b=build/rtl
mkdir -p $b results/rtl
out=results/rtl/tfdpa_f16-$(git rev-parse --short HEAD).jsonl
: > "$out"

build() {  # name, latency, extra verilator flags...
    local name=$1 lat=$2
    shift 2
    verilator --cc --exe --build -O3 -Wall -Wno-UNUSEDSIGNAL --top-module tfdpa_f16 \
        -Mdir "$b/$name" -o tb -CFLAGS "-DTB_LAT=$lat" "$@" \
        "$PWD/rtl/tfdpa_f16.sv" "$PWD/tb/tb_tfdpa_f16.cpp" > "$b/$name.log" 2>&1 \
        || { cat "$b/$name.log"; exit 2; }
}
run() {  # variant, name, latency, extra flags...
    local v=$1 name=$2
    shift
    build "$@"
    r=$("$b/$name/tb" "$b/vectors-$v.txt" 3 2>"$b/$name.err")
    echo "{\"variant\": \"$v\", \"dut\": \"$name\", \"result\": $r}" | tee -a "$out"
}

for spec in "f32.f16.f16.f32 0 0" "f32.f16.f16.f16 1 0" "f16.f16.f16.f16 1 1"; do
    set -- $spec
    v=$1 cf=$2 df=$3
    p="-GC_F16=$cf -GD_F16=$df"
    python tools/rtl_vectors.py --variant "$v" --cases "$n" --seed 1 --out "$b/vectors-$v.txt"
    run "$v" "$v-comb" 0 -GPIPE=0 $p
    run "$v" "$v-pipe" 8 -GPIPE=1 $p
    # Mutants: each must give mismatches, except where noted in tensor-mirror's results
    run "$v" "$v-F22" 8 -GPIPE=1 $p -GF=22
    run "$v" "$v-F24" 8 -GPIPE=1 $p -GF=24
    run "$v" "$v-ROUND_ALIGN" 8 -GPIPE=1 $p +define+MUT_ROUND_ALIGN
    run "$v" "$v-LAT7" 7 -GPIPE=1 $p
    if [ "$cf" = 1 ]; then run "$v" "$v-EZERO19" 8 -GPIPE=1 $p -GEZERO=-19; fi
    if [ "$df" = 1 ]; then run "$v" "$v-RZ_OUT" 8 -GPIPE=1 $p +define+MUT_RZ_OUT; fi
done
