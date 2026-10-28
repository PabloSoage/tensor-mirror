# GEMM jobs through a loop of mma.sync per 16 x 8 tile, the reference of H-029

`tools/gemm_cases.py` writes GEMM jobs, D = C + A B with M x N x K = 64 x 64 x 256, 96 per
type, all from seed 1:
- the first third are uniform raw bits;
- the second have operands within 2 binades of 1;
- the third have operands within 8 binades of 1, which reaches f16's subnormals and
  infinities.

`tiles` cuts each job into 32 output tiles of 16 x 8 over the whole of K, and
`gpu/mma_sync.cu` runs each one as a loop of `steps` instructions, the D of each the C of the
next. That is how a GEMM kernel computes it. `gather` puts the tiles back together, and
`tools/gpu_cases.py check` compares every tile with the integer model chained instruction by
instruction (`tools/run_h029_gpu.sh`).

| GPU | instruction | steps | elements | finite | mismatches with the model |
|---|---|---|---|---|---|
| RTX 2060 Super (`rtx2060s/`) | `m16n8k8.f16.f16.f16.f16` | 32 | 393,216 | 138,613 | 0 |
| | `m16n8k8.f32.f16.f16.f32` | 32 | 393,216 | 262,144 | 0 |

- **Finite, per third:** the raw third is all NaN or infinite after 256 terms. The f16 third
  with 8 binades keeps 7,541 of 131,072 finite, because most of it overflows f16. In f32 the
  second and third thirds are all finite.
- **`compare-rtl-model-*.json`:** `gemm_cases.py compare` of the GPU's D, the D of
  `rtl/tm_pl_array.v` in Verilator (`../../rtl/tm_pl_array-h029-17d8b7f.md`) and the model.
  All three agree in every element.
- **`sha256.txt`:** names the job files and the GPU's D, which are not kept here. The jobs
  are the same bytes for both types and every GPU, except C.
