#!/usr/bin/env bash
# Every T-FDPA mma.sync instruction MMA-Sim models, 10^5 cases each, then F +/- 1 mutants
# on one instruction per family.
set -u
cd "$(dirname "$0")/.."
out=results/xcheck/all-mma-$(git rev-parse --short HEAD).jsonl
mkdir -p results/xcheck
python tools/xcheck_mmasim.py --all --cases 100000 --seed 1 > "$out"
mutant() {  # arch instruction F
    for f in $(($3 - 1)) $(($3 + 1)); do
        python tools/xcheck_mmasim.py --arch "$1" --inst "$2" --cases 100000 --seed 2 --mutate "F=$f" \
            2>/dev/null >> "$out"
    done
}
mutant "Ampere" m16n8k16.f32.bf16.bf16.f32 24
mutant "Ada Lovelace" m16n8k32.f32.e4m3.e4m3.f32 13
mutant "Hopper" m16n8k8.f32.tf32.tf32.f32 25
mutant "RTX Blackwell" m16n8k32.f32.e2m1.e2m1.f32 25
echo "done: $out"
