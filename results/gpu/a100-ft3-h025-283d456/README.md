# A100 (FT3): e_zero with D in f16 (H-025)

`tools/probe_f16_round.py gen Ampere <instruction>`, run on one A100-PCIE-40GB with
`tools/slurm_mma.sbatch` (`gpu/mma_sync.cu` for `sm_80`, CUDA 12.9.1), job of 2026-10-22;
analysed on 2026-10-27. 86 probe tiles per instruction: an exact f16 rounding tie plus a
product 2**-k below it, at a normal and a subnormal level, with three operand kinds. The
generator is deterministic; `sha256.txt` names the case files and the GPU's `d.bin`.

| instruction | e_zero consistent (-14..-45) | F consistent (10..40) | model with -21 (MMA-Sim 54294d4) |
|---|---|---|---|
| m16n8k8.f16.f16.f16.f16 | -20 only | 24 (with -20) | 1 mismatch: subnormal level, k = 20, subnormal x subnormal |
| m16n8k16.f16.f16.f16.f16 | -20 only | 24 (with -20) | the same tile |

`round.json` was written with the model after the change to -20 (`python/tensormirror/arch.py`),
so it shows 0 mismatches. MMA-Sim v2.1.1 has -20 for Ampere with D in f16.
