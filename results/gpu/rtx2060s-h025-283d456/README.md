# RTX 2060 Super (Turing): e_zero with D in f16 (H-025)

`tools/probe_f16_round.py gen Turing m16n8k8.f16.f16.f16.f16`, run on an RTX 2060 Super with
`gpu/mma_sync.cu` for `sm_75` (CUDA 13.3, `tools/gpu_batch.sh run`) on 2026-10-27, and
analysed the same day. 86 probe tiles: an exact f16 rounding tie plus a
product 2**-k below it, at a normal and a subnormal level, with three operand kinds. The
generator is deterministic; `sha256.txt` names the case files and the GPU's `d.bin`.

| instruction | e_zero consistent (-14..-45) | F consistent (10..40) | model with -21 (MMA-Sim 54294d4) |
|---|---|---|---|
| m16n8k8.f16.f16.f16.f16 | -20 only | 24 (with -20) | 1 mismatch: subnormal level, k = 20, subnormal x subnormal |

`round.json` was written with the model after the change to -20 (`python/tensormirror/arch.py`),
so it shows 0 mismatches. MMA-Sim v2.1.1 has -20 for Turing with D in f16.
