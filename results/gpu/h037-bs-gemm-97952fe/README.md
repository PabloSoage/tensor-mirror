# Block-scale GEMM jobs on the RTX 5070 Ti, the reference of H-037

`tools/run_bs_gemm_gpu.sh <root> 96 <inst>...` at `97952fe`, on the RTX 5070 Ti (sm_120a, CUDA
13.3), with 96 jobs of 64 × 64 × 256 per instruction from seed 1 (`tools/bs_gemm.py`).

- **A job's chain:** each 16 × 8 output tile chains 4 block-scale `mma.sync` (K = 64 each)
  in order. C comes from the first instruction, and each D is the next one's C.
- **The scales:** they sit in the lanes' registers under random valid selectors
  (`bs_cases.scale_regs`).
- **The thirds:** as `bs_cases.py gen`:
  - everything raw (scales too, so overflow and NaN are common);
  - raw elements with scales near 1;
  - elements, C and scales near 1.

| instruction | elements | finite per third | mismatches with the chained model |
|---|---|---|---|
| `m16n8k64.mxf4.2x.f32.e2m1.e2m1.f32.ue8m0` (MXFP4) | 393,216 | 39,837 / 130,608 / 131,072 | 0 |
| `m16n8k64.mxf4nvf4.4x.f32.e2m1.e2m1.f32.ue4m3` (NVFP4) | 393,216 | 102,592 / 130,608 / 131,072 | 0 |

- **`check-*.json`:** `tools/bs_cases.py check --steps 4` against the measured model.
- **The RTL:** the 4 × 4 GST array of `rtl/tm_pl_array.v` (GST = 1) gives this D byte for
  byte in Verilator, for both instructions (`../../rtl/tm_pl_array-h037-97952fe.jsonl`).
- **`sha256.txt`:** names the job files and the GPU's D, which are not kept here.
