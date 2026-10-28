# GEMM jobs of 128 x 64 x 256 through a loop of mma.sync, the reference of H-033

As `../h029-gemm-17d8b7f`, with M = 128 so that the 8 x 4 array (blocks of 128 x 4) takes a
job whole: 96 jobs per type from seed 1, in the same thirds (raw bits, ±2 binades, ±8
binades), through `tools/run_h029_gpu.sh Turing 75 8 <root> 96 128`.

| GPU | instruction | steps | elements | finite | mismatches with the model |
|---|---|---|---|---|---|
| RTX 2060 Super | `m16n8k8.f16.f16.f16.f16` | 32 | 786,432 | 276,917 | 0 |
| | `m16n8k8.f32.f16.f16.f32` | 32 | 786,432 | 524,288 | 0 |

- **`compare-rtl-*.json`:** the D of the 8 x 4 `rtl/tm_pl_array.v` in Verilator
  (`../../rtl/tm_pl_array-8x4-h033-4c990e9.jsonl`) against the GPU's: 0 differences in every
  element.
- **`sha256.txt`:** names the job files and the GPU's D, which are not kept here.
