#!/usr/bin/env bash
# Build rtl/gst_fdpa.sv with Verilator for kind::mxf4 / kind::mxf4nvf4 block_scale
# instructions, combinational and pipelined, check it against python/tensormirror/
# blockscale.py, then check that the testbench catches mutants.
#   [XP=1] [SP=1] tools/run_gst_rtl.sh <out.jsonl> <cases> <instruction>...
#   e.g. m16n8k64.mxf4nvf4.4x.f32.e2m1.e2m1.f32.ue4m3
# XP=1 and SP=1 in the environment build the pipelined trees and the split stage 6.
set -u
cd "$(dirname "$0")/.."
out=$1 n=$2
shift 2
b=build/rtl-gst
rtl=$PWD/rtl
mkdir -p $b "$(dirname "$out")"
xp=${XP:-0} sp=${SP:-0}
lat=$((8 + 2 * xp + xp + sp))  # gst_fdpa's LATENCY for K = 64, G = 16: 8 + LG (2) + LS (1) + LN
tag=$([ "$xp" = 1 ] && echo -xp)$([ "$sp" = 1 ] && echo -sp)

for inst in "$@"; do
    nb=$(echo "$inst" | cut -d. -f3 | tr -dc 0-9)
    ue4m3=$([ "${inst##*.}" = ue4m3 ] && echo 1 || echo 0)
    python tools/bs_rtl_vectors.py --inst "$inst" --cases "$n" --seed 1 --out "$b/vectors-$inst.txt" > /dev/null \
        || exit 2
    run() {  # name, latency, extra verilator flags...
        local name=$1 lat=$2
        shift 2
        verilator --cc --exe --build -O3 -Wall -Wno-UNUSEDSIGNAL \
            --top-module gst_fdpa -Mdir "$b/$inst$tag-$name" -o tb \
            -CFLAGS "-DTB_LAT=$lat -DTB_K=64 -DTB_NB=$nb -std=c++17" -GNB=$nb -GUE4M3=$ue4m3 -GXP=$xp -GSP=$sp "$@" \
            "$rtl/pipe_sum.sv" "$rtl/pipe_delay.sv" "$rtl/gst_fdpa.sv" "$PWD/tb/tb_gst_fdpa.cpp" \
            > "$b/$inst$tag-$name.log" 2>&1 \
            || { tail -30 "$b/$inst$tag-$name.log"; echo "{\"config\": \"$inst$tag\", \"dut\": \"$name\", \"build\": \"failed\"}" | tee -a "$out"; return; }
        r=$("$b/$inst$tag-$name/tb" "$b/vectors-$inst.txt" 3 2>"$b/$inst$tag-$name.err")
        echo "{\"config\": \"$inst$tag\", \"dut\": \"$name\", \"result\": $r}" | tee -a "$out"
    }
    run comb 0 -GPIPE=0
    run pipe $lat -GPIPE=1
    [ "${MUTANTS:-1}" = 1 ] || continue
    run F34 $lat -GPIPE=1 -GF=34
    run F36 $lat -GPIPE=1 -GF=36
    run G8 $lat -GPIPE=1 -GG=8
    run ZERO_RULE $lat -GPIPE=1 +define+MUT_GST_ZERO_RULE
    run "LAT$((lat - 1))" $((lat - 1)) -GPIPE=1
done
