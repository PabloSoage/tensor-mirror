#!/usr/bin/env bash
# The Volta cross-check against MMA-Sim: three instructions at 10^6 cases, then mutants
# that must be caught (and the e_zero ones that random bits are not expected to catch).
set -u
cd "$(dirname "$0")/.."
out=results/xcheck/volta-$(git rev-parse --short HEAD).jsonl
mkdir -p results/xcheck
: > "$out"
for t in f32.f16.f16.f32 f32.f16.f16.f16 f16.f16.f16.f16; do
    python tools/xcheck_mmasim.py --arch Volta --inst "m8n8k4.$t" --cases 1000000 --seed 1 | tee -a "$out"
done
for m in "f32.f16.f16.f32 F=22" "f32.f16.f16.f32 F=24" "f32.f16.f16.f32 rounding=rne" \
         "f16.f16.f16.f16 rounding=rz" "f32.f16.f16.f32 e_zero=-130" "f32.f16.f16.f16 e_zero=-19"; do
    set -- $m
    python tools/xcheck_mmasim.py --arch Volta --inst "m8n8k4.$1" --cases 100000 --seed 2 --mutate "$2" \
        | head -1 | tee -a "$out"
done
