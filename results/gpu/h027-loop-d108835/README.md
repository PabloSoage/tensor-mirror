# A loop of mma.sync per tile against the chained model (H-027)

`gpu/mma_sync.cu` with `steps = 16`: each warp runs 16 instructions in a row on its 16 x 8
tile, K after K, with the D of each as the C of the next, in registers. That is a GEMM's inner
loop with K = 16·k. Cases from `tools/gpu_cases.py gen --steps 16`, 1,024 tiles per instruction
(half uniform raw bits, half scaled), plus the directed f16 tiles. `check` chains the model
instruction by instruction (`tools/gpu_cases.py` at `d108835`).

| GPU | build | instructions | elements | mismatches |
|---|---|---|---|---|
| RTX 2060 Super | `sm_75`, CUDA 13.3 | 2 | 264,832 | 0 |
| A100-PCIE-40GB | `sm_80`, CUDA 12.9.1 (`tools/slurm_mma.sbatch`) | 8 | 1,053,952 | 0 |
| RTX 5070 Ti | `sm_120a`, CUDA 13.3 | 74 | 9,704,704 | 0 |

- **The chaining has to be per instruction.** The first check of the 5070 Ti chained the
  whole row in groups of L. Where an instruction's k is below L (k8 f16 and bf16 with
  L = 16, k16 FP8 with L = 32, k4 TF32 with L = 8), that merges two instructions into one
  group. It gave thousands of mismatches per instruction, while the GPU rounds after each
  instruction. On the A100 and the 2060 Super k >= L everywhere, so it made no difference
  there. The `check.json` files here are from the corrected check.
- **`steps1-vs-base.jsonl`:** with `steps = 1` the new kernel's `d.bin` is byte for byte the
  old kernel's (`689d559`) on the same cases, for every instruction of both local builds.
- **`sha256.txt`:** names the case files and the GPU's output, which are not kept here.
