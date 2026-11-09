#!/usr/bin/env bash
# Build rtl/tm_pl_array.v (an R x C tm_array of one instruction) with Verilator and run its
# testbench on the jobs of tools/gemm_cases.py: with random valid/ready, then with neither
# held back (cycles per job), then the SLOT, ALWAYS_READY and SIDE mutants, which must fail.
#   tools/run_array.sh <out.jsonl> <dir> <arch> <D.A.B.C> [R] [C] [Q] [M] [N] [K] [KI]
# KI is the k of the instruction the GPU's loop ran, L by default (H-035). <dir> holds gemm.json, in.bin and expected.bin (gemm_cases.py rtl); the D of the first run
# goes to <dir>/rtl/gemm_d.bin. The units have XP = 1 and SP = 1, as on the board.
set -u
cd "$(dirname "$0")/.."
out=$1 dir=$2 arch=$3 types=$4 R=${5:-4} C=${6:-4} Q=${7:-16} M=${8:-64} N=${9:-64} K=${10:-256} KI=${11:-0}
b=build/array
mkdir -p $b "$(dirname "$out")" "$dir/rtl"
mapfile -t pr < <(python tools/rtl_params.py "$arch" "$types")
g=${pr[0]}
read -r L aw bw <<< "${pr[1]}"
[ "$KI" = 0 ] && KI=$L
x0=$L
[ "$KI" -lt "$L" ] && x0=$((KI * 65536 + L))
tag=$(echo "$arch-$types-${R}x$C-k$KI" | tr ' ' '_')
ew=$(echo "$types" | cut -d. -f1 | tr -dc 0-9)  # D's (and C's) element width
d=0  # tfdpa's LATENCY with XP = 1 and SP = 1: 9 + 2 * ($clog2(L + 1) / 2)
while [ $((1 << d)) -lt $((L + 1)) ]; do d=$((d + 1)); done
lat=$((9 + 2 * (d / 2)))
nin=$(( (M * K * aw + N * K * bw + M * N * ew) / 256 )) nout=$(( M * N * ew / 128 ))
x1=$(( Q * 65536 + R * 256 + C ))
build() {  # name, extra verilator flags...
    local name=$1
    shift
    verilator --cc --exe --build -O3 -Wall -Wno-UNUSEDSIGNAL --top-module tm_pl_array \
        -Mdir "$b/$tag-$name" -o tb $g -GXP=1 -GSP=1 -GR=$R -GC=$C -GQ=$Q -GMJ=$M -GNJ=$N -GKJ=$K -GKI=$KI "$@" \
        -CFLAGS "-DTB_NIN=$nin -DTB_NOUT=$nout -DTB_LAT=$lat -DTB_L=$x0 -DTB_X1=$x1 -DTB_X2=$nin -DTB_X3=$nout -std=c++17" \
        "$PWD/rtl/tm_regs.v" "$PWD/rtl/tm_pl_array.v" "$PWD/rtl/tm_array.sv" \
        "$PWD/rtl/pipe_max.sv" "$PWD/rtl/pipe_sum.sv" "$PWD/rtl/pipe_delay.sv" "$PWD/rtl/tfdpa.sv" \
        "$PWD/tb/tb_tm_pl_array.cpp" > "$b/$tag-$name.log" 2>&1 \
        || { cat "$b/$tag-$name.log"; exit 2; }
}
run() {  # name, D file, seed, p(valid), p(ready)
    r=$("$b/$tag-$1/tb" "$dir/in.bin" "$dir/expected.bin" "$2" "$3" "$4" "$5" 2>"$b/$tag-$1.err")
    echo "{\"config\": \"$tag\", \"dut\": \"$1\", \"p_valid\": $4, \"p_ready\": $5, \"exit\": $?, \"result\": $r}" \
        | tee -a "$out"
}
build tm_pl_array
build SLOT +define+MUT_SLOT
build ALWAYS_READY +define+MUT_ALWAYS_READY
build SIDE +define+MUT_SIDE
run tm_pl_array "$dir/rtl/gemm_d.bin" 3 70 70
run tm_pl_array "$b/$tag-full.bin" 1 100 100
run SLOT "$b/$tag-slot.bin" 3 70 70
run ALWAYS_READY "$b/$tag-ready.bin" 3 70 70
run SIDE "$b/$tag-side.bin" 3 70 70
