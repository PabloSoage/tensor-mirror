#!/usr/bin/env bash
# H-014: rtl/tfdpa.sv against the integer model in every generation without block scaling.
#   [XP=1] tools/run_h014.sh [cases]       results in results/rtl/tfdpa-<commit>[-xp].jsonl
set -u
cd "$(dirname "$0")/.."
out=results/rtl/tfdpa-$(git rev-parse --short HEAD)$([ "${XP:-0}" = 1 ] && echo -xp).jsonl
: > "$out"
bash tools/run_rtl_generic.sh "$out" "${1:-100000}" \
    "Volta|f32.f16.f16.f32|m8n8k4" \
    "Volta|f32.f16.f16.f16|m8n8k4|-20" \
    "Volta|f16.f16.f16.f16|m8n8k4|-20" \
    "Ampere|f32.f16.f16.f32|" \
    "Ampere|f16.f16.f16.f16|" \
    "Ampere|f32.bf16.bf16.f32|" \
    "Ampere|f32.tf32.tf32.f32|" \
    "Ada Lovelace|f32.e4m3.e5m2.f32|" \
    "Hopper|f32.f16.f16.f32|" \
    "RTX Blackwell|f32.e4m3.e4m3.f32|" \
    "RTX Blackwell|f32.e2m1.e3m2.f32|"
