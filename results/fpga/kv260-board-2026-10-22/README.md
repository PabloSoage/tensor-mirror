# KV260 runs of tm_pl_x: H-019, H-020, H-023 and H-024 on the board

Two `run_all.sh` sessions on 2026-10-22, the second for the H-023 directories g1 to g3 (the board's `tools/kv260/load.sh` and
`tools/kv260/tm_run.py` as of `6518aa1`). Per directory, the system was loaded with
`load.sh <BUILD>`, then `tm_run.py` streamed the case files through the PL and wrote
`results*.bin`; on the laptop `tools/kv260/board_cases_x.py check` compared every result with
the vectors (`check*.json`). `run.log` holds the board's output, with the SHA-256 of the image,
the case files and the results.

| dir | system (image in) | GPU | instruction | case files | results | mismatches |
|---|---|---|---|---|---|---|
| h019 | `kv260-1bbf53c` | RTX 5070 Ti | `m16n8k32.mxf8f6f4.1x.f32.e4m3.e4m3.f32.ue8m0` | `cases.bin` | 262,784 | 0 |
| a1 | `kv260-h020/a100-m16n8k8.f32.f16.f16.f32` | A100 | `m16n8k8` (K = L) | `cases.bin` | 525,696 | 0 |
| | | A100 | `m16n8k16.f32.f16.f16.f32`, passes 1 and 2 | `chain-1.bin`, `chain-2.bin` | 2 × 525,696 | 0, 0 |
| a2 | `kv260-h020/a100-m16n8k8.f32.bf16.bf16.f32` | A100 | `m16n8k8` (K = L) | `cases.bin` | 524,288 | 0 |
| | | A100 | `m16n8k16.f32.bf16.bf16.f32`, passes 1 and 2 | `chain-1.bin`, `chain-2.bin` | 2 × 524,288 | 0, 0 |
| a3 | `kv260-h020/a100-m16n8k8.f16.f16.f16.f16` | A100 | `m16n8k8` (K = L) | `cases.bin` | 525,568 | 0 |
| | | A100 | `m16n8k16.f16.f16.f16.f16`, passes 1 and 2 | `chain-1.bin`, `chain-2.bin` | 2 × 525,568 | 0, 0 |
| a4 | `kv260-h020/a100-m16n8k4.f32.tf32.tf32.f32` | A100 | `m16n8k4` (K = L) | `cases.bin` | 524,288 | 0 |
| | | A100 | `m16n8k8.f32.tf32.tf32.f32`, passes 1 and 2 | `chain-1.bin`, `chain-2.bin` | 2 × 524,288 | 0, 0 |
| b1 | `kv260-h020/rtx5070ti-m16n8k32.f32.e4m3.e4m3.f32` | RTX 5070 Ti | `m16n8k32.f32.e4m3.e4m3.f32` | `cases.bin` | 262,144 | 0 |
| g1 | `kv260-h023`, the same instruction | RTX 5070 Ti | `m16n8k64.mxf4.2x.f32.e2m1.e2m1.f32.ue8m0` (MXFP4) | `cases.bin` | 262,784 | 0 |
| g2 | `kv260-h023`, the same instruction | RTX 5070 Ti | `m16n8k64.mxf4nvf4.2x.f32.e2m1.e2m1.f32.ue8m0` | `cases.bin` | 262,784 | 0 |
| g3 | `kv260-h023`, the same instruction | RTX 5070 Ti | `m16n8k64.mxf4nvf4.4x.f32.e2m1.e2m1.f32.ue4m3` (NVFP4) | `cases.bin` | 263,040 | 0 |

In all, 7,613,056 results and 0 mismatches; no result beat had non-zero padding. Chain pass 1
is compared with the model's intermediate result, pass 2 and every `cases.bin` with what the
GPU gave.

- **Clock:** `pl0_ref` read 214,285,713 Hz in every directory (IOPLL / 7).
- **Images:** the SHA-256 of each `tensor-mirror.bit.bin` in `run.log` is the one in the
  build's `sha256.txt`, and `BUILD` in the registers is the build's commit.
- **Case files:** each SHA-256 in `run.log` is that of the file written on the laptop by
  `tools/run_h019.sh`, `tools/run_h020.sh`, `tools/run_h023.sh` or `tools/run_h024.sh` (`build/h019`,
  `build/h020`, `build/h023`, `build/h024`). The vectors and results are not kept here (8 MB each).
- **Throughput:** `tm_run.py` reports about 53 M cases/s with 2 beats per case and 36 M with
  3 (one 64 MB transfer at a time, no stalls). This is the DMA's rate with the PL keeping up,
  not a measure of the unit.
