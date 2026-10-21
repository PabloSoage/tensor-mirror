#!/usr/bin/env bash
# Build rtl/tfdpa.sv with Verilator as one instruction of python/tensormirror/arch.py,
# combinational and pipelined, check it against the integer model, then check that the
# testbench catches mutants (each must give mismatches where its parameter matters).
#   tools/run_rtl_generic.sh <out.jsonl> <cases> "<arch>|<D.A.B.C>|<shape>[|<mutant EZERO>]" ...
# A spec "bs|<block_scale instruction>" is a kind::mxf8f6f4 instruction (ST = 1), with the
# vectors of tools/bs_rtl_vectors.py. XP=1 in the environment builds the pipelined trees,
# SP=1 the split stage 6.
set -u
cd "$(dirname "$0")/.."
out=$1 n=$2
shift 2
b=build/rtl-generic
mkdir -p $b "$(dirname "$out")"

for spec in "$@"; do
    IFS="|" read -r arch types shape ezmut <<< "$spec"
    tag=$(echo "$arch-$types-$shape" | tr ' ' '_' | sed 's/-$//')
    st=0
    if [ "$arch" = bs ]; then
        st=1 tag=$types
        mapfile -t pr < <(python tools/rtl_params.py bs "$types")
        python tools/bs_rtl_vectors.py --inst "$types" --cases "$n" --seed 1 --out "$b/vectors-$tag.txt" > /dev/null
    else
        mapfile -t pr < <(python tools/rtl_params.py "$arch" "$types" "$shape")
        python tools/rtl_vectors.py --arch "$arch" --variant "$types" --shape "$shape" \
            --cases "$n" --seed 1 --out "$b/vectors-$tag.txt" > /dev/null
    fi
    vtag=$tag  # the vectors keep this name; with XP and SP the results get "-xp", "-sp"
    g=${pr[0]}
    read -r L aw bw <<< "${pr[1]}"
    run() {  # name, latency, extra verilator flags...
        local name=$1 lat=$2
        shift 2
        verilator --cc --exe --build -O3 -Wall -Wno-UNUSEDSIGNAL \
            --top-module tfdpa -Mdir "$b/$tag-$name" -o tb \
            -CFLAGS "-DTB_LAT=$lat -DTB_L=$L -DTB_AW=$aw -DTB_BW=$bw -DTB_ST=$st -std=c++17" $g "$@" \
            "$PWD/rtl/pipe_max.sv" "$PWD/rtl/pipe_sum.sv" "$PWD/rtl/pipe_delay.sv" \
            "$PWD/rtl/tfdpa.sv" "$PWD/tb/tb_tfdpa.cpp" > "$b/$tag-$name.log" 2>&1 \
            || { tail -30 "$b/$tag-$name.log"; echo "{\"config\": \"$tag\", \"dut\": \"$name\", \"build\": \"failed\"}" | tee -a "$out"; return; }
        r=$("$b/$tag-$name/tb" "$b/vectors-$vtag.txt" 3 2>"$b/$tag-$name.err")
        echo "{\"config\": \"$tag\", \"dut\": \"$name\", \"result\": $r}" | tee -a "$out"
    }
    F=$(echo "$g" | grep -o 'GF=[0-9]*' | cut -d= -f2)
    rne=$(echo "$g" | grep -o 'GRNE=[0-9]' | cut -d= -f2)
    xp=${XP:-0}
    lat=8
    if [ "$xp" = 1 ]; then  # tfdpa's LATENCY: 8 + 2 * ($clog2(L + 1) / 2)
        d=0
        while [ $((1 << d)) -lt $((L + 1)) ]; do d=$((d + 1)); done
        lat=$((8 + 2 * (d / 2)))
        tag=$tag-xp
    fi
    sp=${SP:-0}
    if [ "$sp" = 1 ]; then lat=$((lat + 1)) tag=$tag-sp; fi
    p="-GPIPE=1 -GXP=$xp -GSP=$sp"
    run comb 0 -GPIPE=0 -GXP=$xp -GSP=$sp
    run pipe $lat $p
    [ "${MUTANTS:-1}" = 1 ] || continue
    run "F$((F - 1))" $lat $p -GF=$((F - 1))
    run "F$((F + 1))" $lat $p -GF=$((F + 1))
    run ROUND_ALIGN $lat $p +define+MUT_ROUND_ALIGN
    run "LAT$((lat - 1))" $((lat - 1)) $p
    if [ "$rne" = 1 ]; then run RZ_OUT $lat $p +define+MUT_RZ_OUT; fi
    if [ -n "${ezmut:-}" ]; then run "EZERO$ezmut" $lat $p -GEZERO=$ezmut; fi
    if [ "$st" = 1 ]; then run ST_NOSCALE $lat $p +define+MUT_ST_NOSCALE; fi
done
