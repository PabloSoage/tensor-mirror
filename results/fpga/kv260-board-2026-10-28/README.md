# KV260 runs of tm_pl_x: H-025 on the board

One `run_all.sh` session on 2026-10-28 (the board's `tools/kv260/load.sh` and
`tools/kv260/tm_run.py` as on 2026-10-22). Same procedure as `../kv260-board-2026-10-22`: load
the system with `load.sh <BUILD>`, stream the case files through the PL, then compare every
result with the vectors using `tools/kv260/board_cases_x.py check` (`check*.json`).

| dir | system (image in) | EZERO | case files | results | mismatches |
|---|---|---|---|---|---|
| a3n | `kv260-h025/a100-m16n8k8.f16.f16.f16.f16` | -20 | `cases.bin` (`m16n8k8`, K = L) | 525,568 | 0 |
| | | | `chain-1.bin`, `chain-2.bin` (`m16n8k16`, passes 1 and 2) | 2 × 525,568 | 0, 0 |
| | | | `chain-probe.bin` (`tools/probe_f16_round.py`, A100) | 11,008 | 0 |
| a3old-probe | `kv260-h020/a100-m16n8k8.f16.f16.f16.f16` | -21 | `chain-probe.bin` | 11,008 | 1 |

- **Vectors:** `build/h025/vec/{a3,a3-chain-1,a3-chain-2,a3-probe}.txt`, from the model at
  `60213eb` with the GPU's outputs (L = 8, no scales). `cases.bin`, `chain-1.bin` and
  `chain-2.bin` are byte for byte those of 2026-10-22.
- **The control:** the H-020 image, with -21, fails on one probe case, 10496 (subnormal level,
  k = 20, subnormal × subnormal). The board gives `0x1` and the A100 gives `0x0`. This is the
  case where Verilator and the model with -21 also fail (`../../rtl/tm_pl_x-h025-60213eb.md`).
- **`a3old-probe/run.log`:** this directory held only the probe, so `tm_run.py cases.bin`
  failed (`rc=1`). Its traceback, which named the board's home path, has been removed from the log.
- **Images:** the SHA-256 of each `tensor-mirror.bit.bin` in `run.log` is the one in the
  build's `sha256.txt`, and `BUILD` in the registers is the build's commit (`60213EB9`,
  `20FAF52A`).
- **Clock:** `pl0_ref` read 214,285,713 Hz (IOPLL / 7). Throughput was about 53 M cases/s with
  no stalls, as before.

## The 4 x 4 array (H-029), later the same day

A second session, from a directory of its own: `run_all.sh`, `load.sh` and `tm_run.py` as of
`81b2e45`, with the overlay's `assigned-clock-rates` at 187,500,000, because the array does
not close at 214.29 MHz (`../kv260-h029`). `tm_run.py` streams the 96 jobs through the array
in one DMA transfer and writes their D.

| dir | system (image in) | jobs | elements of D | mismatches with the RTX 2060 Super |
|---|---|---|---|---|
| arr16 | `kv260-h029/a100-4x4-m64n64k256.f16.f16.f16.f16` | 96 | 393,216 | 0 |

- **The D is the GPU's, byte for byte.** The SHA-256 of `results.bin` in `arr16/run.log`,
  `01221d02…`, is that of the RTX 2060 Super's D in `../../gpu/h029-gemm-17d8b7f/rtx2060s/sha256.txt`.
  `arr16/compare.json` counts the same element by element.
- **The case file** is `tools/gemm_cases.py rtl` of those jobs, A, B^T and C of each one packed.
- **Image, build and clock:** the image's SHA-256 is the build's, `BUILD` reads `17D8B7F6`,
  and `pl0_ref` read 187,499,999 Hz.
- **Throughput:** 96 jobs in 0.007 s of DMA, about 14,200 jobs per second, or 30 GFLOP/s
  (2·64·64·256 per job). That is about 13,200 cycles per job at 187.5 MHz, against 11,027 in
  Verilator with no gaps. A single transfer this short is a rough measure.

## The array's later builds (H-029 f32, H-030, H-031, H-032), the same evening

The same jobs, the files from `81b2e45` on (no change to `load.sh` or `tm_run.py` since).
The 187.5 MHz runs come from the directory of `arr16`, and the 214.29 MHz runs from one whose
overlay asks for 200 MHz, as `fpga/kv260/tensor-mirror.dtsi` does.

| dir | system (image in) | clock | jobs | SHA-256 of results.bin | jobs per second |
|---|---|---|---|---|---|
| arr32 | `kv260-h029/a100-4x4-m64n64k256.f32.f16.f16.f32` | 187.5 MHz | 96 | `ab8fb251…` | 13,202 |
| b32 | `kv260-h031/a100-4x4-m64n64k256.f32.f16.f16.f32` | 187.5 MHz | 96 | `ab8fb251…` | 13,196 |
| a16 | `kv260-h030/a100-4x4-m64n64k256.f16.f16.f16.f16` | 214.29 MHz | 96 | `01221d02…` | 16,238 |
| b16 | `kv260-h031/a100-4x4-m64n64k256.f16.f16.f16.f16` | 214.29 MHz | 96 | `01221d02…` | 16,236 |
| c32 | `kv260-h032/a100-4x4-m64n64k256.f32.f16.f16.f32` | 214.29 MHz | 96 | `ab8fb251…` | 15,070 |
| c16 | `kv260-h032/a100-4x4-m64n64k256.f16.f16.f16.f16` | 214.29 MHz | 96 | `01221d02…` | 16,237 |

- **Every output is the GPUs' D, byte for byte.** `01221d02…` is the f16 D and `ab8fb251…` the
  f32 D of both the RTX 2060 Super and the A100 (`../../gpu/h029-gemm-17d8b7f/*/sha256.txt`).
  That is 786,432 elements per type.
- **Images, builds and clocks:** each image's SHA-256 in `run.log` is its build's, `BUILD`
  reads the build's commit, and `pl0_ref` read 187,499,999 or 214,285,713 Hz.
- **At 214.29 MHz:** 16,238 jobs per second in f16 is 34 GFLOP/s, and 15,070 in f32 is
  32 GFLOP/s. Both are rough, from a single transfer of 6 ms.

## The 8 x 4 array (H-033), later still

The 187.5 MHz directory, with the jobs of `../../gpu/h033-gemm-4c990e9` (128 x 64 x 256).

| dir | system (image in) | clock | jobs | SHA-256 of results.bin | jobs per second |
|---|---|---|---|---|---|
| x16 | `kv260-h033/a100-8x4-m128n64k256.f16.f16.f16.f16` | 187.5 MHz | 96 | `e16863b5…` | 11,532 |
| x32 | `kv260-h033/a100-8x4-m128n64k256.f32.f16.f16.f32` | 187.5 MHz | 96 | `8eafccc7…` | 10,237 |

- **Both outputs are the RTX 2060 Super's D, byte for byte**
  (`../../gpu/h033-gemm-4c990e9/rtx2060s/sha256.txt`), 786,432 elements per type.
- **Throughput:** about 48 GFLOP/s in f16 and 43 in f32 (2·128·64·256 per job). In f16 that is
  about 16,260 cycles per job, against 12,820 in Verilator with no gaps. The difference is
  the DMA's, and loading still does not overlap computing.
- The images' SHA-256 are their builds', `BUILD` reads `4C990E9B`, and `pl0_ref` read
  187,499,999 Hz.
