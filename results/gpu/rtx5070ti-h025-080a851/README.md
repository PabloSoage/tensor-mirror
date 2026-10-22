# RTX 5070 Ti: e_zero with D in f16 (H-025)

`tools/probe_f16_round.py gen "RTX Blackwell" m16n8k16.f16.f16.f16.f16`, run with
`gpu/mma_sync.cu` built for `sm_120a` (CUDA 13.0, `gpu_batch.sh run`), then `analyze`.
86 probe tiles: an exact f16 rounding tie plus a product 2**-k below it, at a normal and a
subnormal level, with three operand kinds.

| instruction | model (F, e_zero) | mismatches | F consistent | e_zero consistent (-14..-45) |
|---|---|---|---|---|
| m16n8k16.f16.f16.f16.f16 | 25, -21 | 0 | 25 | -21 only |

MMA-Sim v2.0.3, v2.1.0 and v2.1.1 all have -21 for RTX Blackwell with D in f16.
