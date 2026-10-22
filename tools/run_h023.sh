#!/usr/bin/env bash
# H-023, laptop side: rtl/tm_pl_x.v with GST = 1 in Verilator (tools/run_pl_x.sh) for the GST
# instructions the RTX 5070 Ti ran in H-010, against the model and against the GPU; tm_pl_x with
# GST = 0 again as H-019's and one of H-020's instructions; and the board's case files.
#   tools/run_h023.sh <H-010 run dir: holds one directory per instruction> [model cases]
# Results in results/rtl/tm_pl_x-h023-<commit>.jsonl; vectors and case files in build/h023/.
set -u
cd "$(dirname "$0")/.."
g=$1 n=${2:-100000}
out=results/rtl/tm_pl_x-h023-$(git rev-parse --short HEAD).jsonl
: > "$out"
mkdir -p build/h023
bash tools/run_pl_x.sh "$out" "$n" "bs|m16n8k32.mxf8f6f4.1x.f32.e4m3.e4m3.f32.ue8m0" || exit 2
bash tools/run_pl_x.sh "$out" "$n" "Ampere|f32.f16.f16.f32|m16n8k8" || exit 2
for inst in m16n8k64.mxf4.2x.f32.e2m1.e2m1.f32.ue8m0 m16n8k64.mxf4nvf4.2x.f32.e2m1.e2m1.f32.ue8m0 \
            m16n8k64.mxf4nvf4.4x.f32.e2m1.e2m1.f32.ue4m3; do
    ns=$(echo "$inst" | cut -d. -f3 | tr -dc 0-9)
    bash tools/run_pl_x.sh "$out" "$n" "gst|$inst" || exit 2
    python tools/kv260/board_cases_x.py gpu "$g/$inst" "build/h023/$inst.txt" || exit 2
    r=$(build/pl-x/$inst-tm_pl_x/tb "build/h023/$inst.txt" 5 70 70 2> "build/h023/$inst.err")
    echo "{\"config\": \"$inst\", \"dut\": \"tm_pl_x\", \"vectors\": \"rtx5070ti-h010\", \"exit\": $?, \"result\": $r}" \
        | tee -a "$out"
    python tools/kv260/board_cases_x.py pack "build/h023/$inst.txt" 64 4 4 "$ns" "build/h023/$inst.bin" || exit 2
done
sha256sum build/h023/*.bin
