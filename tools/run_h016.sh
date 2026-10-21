#!/usr/bin/env bash
# H-016: the block-scaled RTL against python/tensormirror/blockscale.py: the 25 kind::mxf8f6f4
# pairs (rtl/tfdpa.sv, ST = 1) and the kind::mxf4 / mxf4nvf4 instructions (rtl/gst_fdpa.sv).
#   tools/run_h016.sh [cases]       results in results/rtl/blockscale-<commit>.jsonl
set -u
cd "$(dirname "$0")/.."
out=results/rtl/blockscale-$(git rev-parse --short HEAD).jsonl
: > "$out"
specs=()
for a in e4m3 e5m2 e3m2 e2m3 e2m1; do
    for b in e4m3 e5m2 e3m2 e2m3 e2m1; do
        specs+=("bs|m16n8k32.mxf8f6f4.1x.f32.$a.$b.f32.ue8m0")
    done
done
bash tools/run_rtl_generic.sh "$out" "${1:-100000}" "${specs[@]}"
bash tools/run_gst_rtl.sh "$out" "${1:-100000}" \
    m16n8k64.mxf4.2x.f32.e2m1.e2m1.f32.ue8m0 \
    m16n8k64.mxf4nvf4.2x.f32.e2m1.e2m1.f32.ue8m0 \
    m16n8k64.mxf4nvf4.4x.f32.e2m1.e2m1.f32.ue4m3 \
    m16n8k64.mxf4nvf4.4x.f32.e2m1.e2m1.f32.ue8m0
