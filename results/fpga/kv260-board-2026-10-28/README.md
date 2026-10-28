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
