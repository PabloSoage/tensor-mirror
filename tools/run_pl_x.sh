#!/usr/bin/env bash
# Build rtl/tm_pl_x.v (one tfdpa or gst_fdpa unit behind AXI4-Stream) with Verilator as one
# instruction and run its testbench, with random valid/ready, on the cases of
# tools/rtl_vectors.py or tools/bs_rtl_vectors.py; then check that the ALWAYS_READY mutant is
# caught.
#   tools/run_pl_x.sh <out.jsonl> <cases> "<arch>|<D.A.B.C>|<shape>" | "bs|<instruction>" | "gst|<instruction>"
# The unit has XP = 1 and SP = 1, as on the board.
set -u
cd "$(dirname "$0")/.."
out=$1 n=$2 spec=$3
b=build/pl-x
mkdir -p $b "$(dirname "$out")"
IFS="|" read -r arch types shape <<< "$spec"
tag=$(echo "$arch-$types-$shape" | tr ' ' '_' | sed 's/-$//')
st=0 gst=0 ns=1
if [ "$arch" = bs ] || [ "$arch" = gst ]; then
    tag=$types
    mapfile -t pr < <(python tools/rtl_params.py "$arch" "$types")
    python tools/bs_rtl_vectors.py --inst "$types" --cases "$n" --seed 1 --out "$b/vectors-$tag.txt" \
        > /dev/null || exit 2
    if [ "$arch" = bs ]; then st=1; else gst=1 ns=$(echo "$types" | cut -d. -f3 | tr -dc 0-9); fi
else
    mapfile -t pr < <(python tools/rtl_params.py "$arch" "$types" "$shape")
    python tools/rtl_vectors.py --arch "$arch" --variant "$types" --shape "$shape" \
        --cases "$n" --seed 1 --out "$b/vectors-$tag.txt" > /dev/null || exit 2
fi
g=${pr[0]}
read -r L aw bw <<< "${pr[1]}"
if [ "$gst" = 1 ]; then
    lat=12  # gst_fdpa's LATENCY with XP = 1 and SP = 1, K = 64, G = 16
else
    d=0  # tfdpa's LATENCY with XP = 1 and SP = 1: 9 + 2 * ($clog2(L + 1) / 2)
    while [ $((1 << d)) -lt $((L + 1)) ]; do d=$((d + 1)); done
    lat=$((9 + 2 * (d / 2)))
fi
build() {  # name, extra verilator flags...
    local name=$1
    shift
    verilator --cc --exe --build -O3 -Wall -Wno-UNUSEDSIGNAL --top-module tm_pl_x \
        -Mdir "$b/$tag-$name" -o tb $g -GXP=1 -GSP=1 "$@" \
        -CFLAGS "-DTB_LAT=$lat -DTB_L=$L -DTB_AW=$aw -DTB_BW=$bw -DTB_ST=$st -DTB_GST=$gst -DTB_NS=$ns -std=c++17" \
        "$PWD/rtl/tm_regs.v" "$PWD/rtl/tm_pl_x.v" \
        "$PWD/rtl/pipe_max.sv" "$PWD/rtl/pipe_sum.sv" "$PWD/rtl/pipe_delay.sv" "$PWD/rtl/tfdpa.sv" \
        "$PWD/rtl/gst_fdpa.sv" "$PWD/tb/tb_tm_pl_x.cpp" > "$b/$tag-$name.log" 2>&1 \
        || { cat "$b/$tag-$name.log"; exit 2; }
}
run() {  # name
    r=$("$b/$tag-$1/tb" "$b/vectors-$tag.txt" 3 70 70 2>"$b/$tag-$1.err")
    echo "{\"config\": \"$tag\", \"dut\": \"$1\", \"exit\": $?, \"result\": $r}" | tee -a "$out"
}
build tm_pl_x
build ALWAYS_READY +define+MUT_ALWAYS_READY
run tm_pl_x
run ALWAYS_READY
