# GEMM jobs on the RTX 5070 Ti with k < L and k = L, the reference of H-035

The jobs of `../h029-gemm-17d8b7f`: 96 jobs per type of 64 × 64 × 256 from seed 1, in the
same thirds (raw bits, ±2 binades, ±8 binades). Each job goes through a loop of `mma.sync`
that walks K in order, in two versions per type:

- one with an instruction shorter than the unit's L;
- one with an instruction as long as L.

The command was `tools/run_h029_gpu.sh "RTX Blackwell" 120a <k> <root> 96 64 "<types>"`, on
the RTX 5070 Ti (sm_120a), with CUDA 13.3.

| types | L | instruction | steps | mismatches with the model |
|---|---|---|---|---|
| `f16.f16.f16.f16` | 16 | `m16n8k8` | 32 | 0 |
| | | `m16n8k16` | 16 | 0 |
| `f32.f16.f16.f32` | 16 | `m16n8k8` | 32 | 0 |
| | | `m16n8k16` | 16 | 0 |
| `f32.e4m3.e4m3.f32` | 32 | `m16n8k16` | 16 | 0 |
| | | `m16n8k32` | 8 | 0 |
| `f16.e4m3.e4m3.f16` | 32 | `m16n8k16` | 16 | 0 |
| | | `m16n8k32` | 8 | 0 |

Each row has 393,216 elements. The model is the measured integer model, chained one
instruction after another (`summary.jsonl`, `check-*.json`).

- **`compare-k*-k*-*.json`:** the two loops of each type against each other. Same inputs,
  different k: they differ in most elements. In the ±2-binade third:
  - 74 % in `f16.f16.f16.f16`;
  - 93 % in `f32.f16.f16.f32`;
  - 9 % in `f32.e4m3.e4m3.f32`;
  - 62 % in `f16.e4m3.e4m3.f16`.

  The k of the instruction is part of the arithmetic.
- **`compare-rtl-*.json`:** the D of `rtl/tm_pl_array.v` in Verilator, with KI set to the
  instruction's k, against the GPU's. There are 0 differences in all eight cases. The f16
  types use a 4 × 4 array and FP8 a 2 × 4 array. See
  `../../rtl/tm_pl_array-h035-159f42a.jsonl`, whose `cross` runs (KI = L against the k < L
  loop) all mismatch, as they must.
- **`sha256.txt`:** names the job files and the GPU's D, which are not kept here.
