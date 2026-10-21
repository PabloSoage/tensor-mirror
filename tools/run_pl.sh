#!/usr/bin/env bash
# Build rtl/tm_pl.v (three pipelined tfdpa_f16 units behind AXI4-Stream) with Verilator
# and run its testbench on the cases of tools/pl_vectors.py, one set per source variant,
# with random valid/ready; then check that the ALWAYS_READY mutant is caught.
#   tools/run_pl.sh [cases]
set -u
cd "$(dirname "$0")/.."
n=${1:-100000}
b=build/pl
mkdir -p $b
build() {  # name, extra verilator flags...
    local name=$1
    shift
    verilator --cc --exe --build -O3 -Wall -Wno-UNUSEDSIGNAL --top-module tm_pl \
        -Mdir "$b/$name" -o tb "$@" "$PWD/rtl/tm_regs.v" "$PWD/rtl/tm_pl.v" \
        "$PWD/rtl/pipe_max.sv" "$PWD/rtl/pipe_sum.sv" "$PWD/rtl/pipe_delay.sv" "$PWD/rtl/tfdpa.sv" \
        "$PWD/rtl/tfdpa_f16.sv" "$PWD/tb/tb_tm_pl.cpp" > "$b/$name.log" 2>&1 || { cat "$b/$name.log"; exit 2; }
}
build tm_pl
build tm_pl-ALWAYS_READY +define+MUT_ALWAYS_READY
for v in f32.f16.f16.f32 f32.f16.f16.f16 f16.f16.f16.f16; do
    python tools/pl_vectors.py --source "$v" --cases "$n" --seed 1 --out "$b/vectors-$v.txt" || exit 2
    r=$("$b/tm_pl/tb" "$b/vectors-$v.txt" 3 70 70 2>&1)
    s=$?
    echo "$v: $(echo "$r" | tail -1) (exit $s)"
done
echo "ALWAYS_READY mutant: $("$b/tm_pl-ALWAYS_READY/tb" "$b/vectors-f32.f16.f16.f32.txt" 3 70 70 2>&1 | tail -1)"
