#!/usr/bin/env bash
# Build rtl/gst_fdpa.sv with Verilator for kind::mxf4 / kind::mxf4nvf4 block_scale
# instructions, combinational and pipelined, check it against python/tensormirror/
# blockscale.py, then check that the testbench catches mutants.
#   tools/run_gst_rtl.sh <out.jsonl> <cases> <instruction>...
#   e.g. m16n8k64.mxf4nvf4.4x.f32.e2m1.e2m1.f32.ue4m3
set -u
cd "$(dirname "$0")/.."
out=$1 n=$2
shift 2
b=build/rtl-gst
mkdir -p $b "$(dirname "$out")"

for inst in "$@"; do
    nb=$(echo "$inst" | cut -d. -f3 | tr -dc 0-9)
    ue4m3=$([ "${inst##*.}" = ue4m3 ] && echo 1 || echo 0)
    python tools/bs_rtl_vectors.py --inst "$inst" --cases "$n" --seed 1 --out "$b/vectors-$inst.txt" > /dev/null
    run() {  # name, latency, extra verilator flags...
        local name=$1 lat=$2
        shift 2
        verilator --cc --exe --build -O3 -Wall -Wno-UNUSEDSIGNAL \
            --top-module gst_fdpa -Mdir "$b/$inst-$name" -o tb \
            -CFLAGS "-DTB_LAT=$lat -DTB_K=64 -DTB_NB=$nb -std=c++17" -GNB=$nb -GUE4M3=$ue4m3 "$@" \
            "$PWD/rtl/gst_fdpa.sv" "$PWD/tb/tb_gst_fdpa.cpp" > "$b/$inst-$name.log" 2>&1 \
            || { tail -30 "$b/$inst-$name.log"; echo "{\"config\": \"$inst\", \"dut\": \"$name\", \"build\": \"failed\"}" | tee -a "$out"; return; }
        r=$("$b/$inst-$name/tb" "$b/vectors-$inst.txt" 3 2>"$b/$inst-$name.err")
        echo "{\"config\": \"$inst\", \"dut\": \"$name\", \"result\": $r}" | tee -a "$out"
    }
    run comb 0 -GPIPE=0
    run pipe 8 -GPIPE=1
    [ "${MUTANTS:-1}" = 1 ] || continue
    run F34 8 -GPIPE=1 -GF=34
    run F36 8 -GPIPE=1 -GF=36
    run G8 8 -GPIPE=1 -GG=8
    run ZERO_RULE 8 -GPIPE=1 +define+MUT_GST_ZERO_RULE
    run LAT7 7 -GPIPE=1
done
