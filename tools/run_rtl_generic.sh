#!/usr/bin/env bash
# Build rtl/tfdpa.sv with Verilator as one instruction of python/tensormirror/arch.py,
# combinational and pipelined, check it against the integer model, then check that the
# testbench catches mutants (each must give mismatches where its parameter matters).
#   tools/run_rtl_generic.sh <out.jsonl> <cases> "<arch>|<D.A.B.C>|<shape>[|<mutant EZERO>]" ...
set -u
cd "$(dirname "$0")/.."
out=$1 n=$2
shift 2
b=build/rtl-generic
mkdir -p $b "$(dirname "$out")"

for spec in "$@"; do
    IFS="|" read -r arch types shape ezmut <<< "$spec"
    tag=$(echo "$arch-$types-$shape" | tr ' ' '_' | sed 's/-$//')
    mapfile -t pr < <(python tools/rtl_params.py "$arch" "$types" "$shape")
    g=${pr[0]}
    read -r L aw bw <<< "${pr[1]}"
    python tools/rtl_vectors.py --arch "$arch" --variant "$types" --shape "$shape" \
        --cases "$n" --seed 1 --out "$b/vectors-$tag.txt" > /dev/null
    run() {  # name, latency, extra verilator flags...
        local name=$1 lat=$2
        shift 2
        verilator --cc --exe --build -O3 -Wall -Wno-UNUSEDSIGNAL \
            --top-module tfdpa -Mdir "$b/$tag-$name" -o tb \
            -CFLAGS "-DTB_LAT=$lat -DTB_L=$L -DTB_AW=$aw -DTB_BW=$bw -std=c++17" $g "$@" \
            "$PWD/rtl/tfdpa.sv" "$PWD/tb/tb_tfdpa.cpp" > "$b/$tag-$name.log" 2>&1 \
            || { tail -30 "$b/$tag-$name.log"; echo "{\"config\": \"$tag\", \"dut\": \"$name\", \"build\": \"failed\"}" | tee -a "$out"; return; }
        r=$("$b/$tag-$name/tb" "$b/vectors-$tag.txt" 3 2>"$b/$tag-$name.err")
        echo "{\"config\": \"$tag\", \"dut\": \"$name\", \"result\": $r}" | tee -a "$out"
    }
    F=$(echo "$g" | grep -o 'GF=[0-9]*' | cut -d= -f2)
    rne=$(echo "$g" | grep -o 'GRNE=[0-9]' | cut -d= -f2)
    run comb 0 -GPIPE=0
    run pipe 8 -GPIPE=1
    [ "${MUTANTS:-1}" = 1 ] || continue
    run "F$((F - 1))" 8 -GPIPE=1 -GF=$((F - 1))
    run "F$((F + 1))" 8 -GPIPE=1 -GF=$((F + 1))
    run ROUND_ALIGN 8 -GPIPE=1 +define+MUT_ROUND_ALIGN
    run LAT7 7 -GPIPE=1
    if [ "$rne" = 1 ]; then run RZ_OUT 8 -GPIPE=1 +define+MUT_RZ_OUT; fi
    if [ -n "${ezmut:-}" ]; then run "EZERO$ezmut" 8 -GPIPE=1 -GEZERO=$ezmut; fi
done
