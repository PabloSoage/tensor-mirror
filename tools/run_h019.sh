#!/usr/bin/env bash
# H-019, laptop side: rtl/tm_pl_x.v in Verilator (tools/run_pl_x.sh) as the board's instruction
# and as Volta's f16 one, then against what the RTX 5070 Ti gave in H-010 for that instruction,
# and the board's case file.
#   tools/run_h019.sh <H-010 run dir of the instruction> [cases]
# Results in results/rtl/tm_pl_x-<commit>.jsonl; the GPU's vectors and the case file in
# build/h019/.
set -u
cd "$(dirname "$0")/.."
gpu=$1 n=${2:-100000}
inst=m16n8k32.mxf8f6f4.1x.f32.e4m3.e4m3.f32.ue8m0
out=results/rtl/tm_pl_x-$(git rev-parse --short HEAD).jsonl
: > "$out"
bash tools/run_pl_x.sh "$out" "$n" "bs|$inst" || exit 2
bash tools/run_pl_x.sh "$out" "$n" "Volta|f16.f16.f16.f16|m8n8k4" || exit 2
mkdir -p build/h019
python tools/kv260/board_cases_x.py gpu "$gpu" build/h019/gpu-vectors.txt || exit 2
r=$(build/pl-x/$inst-tm_pl_x/tb build/h019/gpu-vectors.txt 5 70 70 2> build/h019/gpu-vectors.err)
echo "{\"config\": \"$inst\", \"dut\": \"tm_pl_x\", \"vectors\": \"rtx5070ti-h010\", \"exit\": $?, \"result\": $r}" \
    | tee -a "$out"
read -r L aw bw < <(python tools/rtl_params.py bs "$inst" | tail -1)
python tools/kv260/board_cases_x.py pack build/h019/gpu-vectors.txt "$L" "$aw" "$bw" 1 build/h019/cases.bin \
    || exit 2
sha256sum build/h019/gpu-vectors.txt build/h019/cases.bin
