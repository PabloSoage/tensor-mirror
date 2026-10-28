# A loop of block-scale mma.sync per tile against the chained model (H-028)

The block-scale harness, `gpu/mma_block_scale.cu`, run with `steps = 8` on the RTX 5070 Ti
(`sm_120a`, CUDA 13.3, `tools/run_h028.sh`). Output tile t chains input tiles 8t .. 8t+7,
each with its own A, B, scale factors and selectors. C comes from the first input tile, and
the f32 D of each step is the C of the next, in registers.

- **Cases:** `tools/bs_cases.py gen` with 4,096 input tiles per instruction, in thirds: raw
  bits, raw elements with scales near 1, and everything near 1, plus the directed tiles. That
  gives 512 output tiles.
- **Check:** `tools/bs_cases.py check --steps 8`, the integer model chained the same way.

| instructions | output elements | finite | finite, chains inside the near-1 third | mismatches |
|---|---|---|---|---|
| 25 `mxf8f6f4` (every FP8/FP6/FP4 pair, ue8m0) | 1,638,400 | 821,537 | 544,000 | 0 |
| `mxf4.2x` ue8m0 (MXFP4) | 65,536 | 45,537 | 21,760 | 0 |
| `mxf4nvf4.2x` ue8m0 | 65,536 | 45,537 | 21,760 | 0 |
| `mxf4nvf4.4x` ue4m3 (NVFP4) | 65,536 | 56,667 | 21,760 | 0 |

- **What NaN does here:** after eight steps of raw bits, most outputs of the first two thirds
  are NaN, because one NaN anywhere in the chain propagates. The model matches those too,
  but they say little. The finite outputs, and above all the chains of near-1 values, are
  what tests the arithmetic. `finite.jsonl` gives the counts per instruction.
- **`steps1-vs-base.jsonl`:** with `steps = 1` the new kernel's `d.bin` is byte for byte the
  old kernel's (`22eee12`) on the same cases, in all 28 instructions.
- **`sha256.txt`:** names the case files and the GPU's output, which are not kept here.
