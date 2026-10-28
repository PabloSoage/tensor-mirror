# tm_pl_array 4 x 4, Ampere f16, in Verilator (H-029)

`rtl/tm_pl_array.v` with R = C = 4 `tfdpa` units, Q = 16 slots each, XP = SP = 1, and one
GEMM job of 64 x 64 x 256 per build (`tools/run_array.sh`, Verilator 5.032). The inputs are
the 96 jobs per type of `../gpu/h029-gemm-17d8b7f/`, and the expected D is the RTX 2060
Super's.

| types | run | p(valid), p(ready) | output beats | mismatched beats | cycles per job |
|---|---|---|---|---|---|
| `f16.f16.f16.f16` | the array | 70 %, 70 % | 49,152 | 0 | 12,236 |
| | the array | 100 %, 100 % | 49,152 | 0 | 11,027 |
| | `MUT_SLOT` | 70 %, 70 % | 49,152 | 32,739 | |
| | `MUT_ALWAYS_READY` | 70 %, 70 % | 13,312 | 8,706 | |
| `f32.f16.f16.f32` | the array | 70 %, 70 % | 98,304 | 0 | 13,331 |
| | the array | 100 %, 100 % | 98,304 | 0 | 11,795 |
| | `MUT_SLOT` | 70 %, 70 % | 98,304 | 65,536 | |
| | `MUT_ALWAYS_READY` | 70 %, 70 % | 27,648 | 11,624 | |

- **Both mutants are caught.** `MUT_SLOT` feeds back another slot's d. `MUT_ALWAYS_READY`
  takes beats while the array runs, so they are lost and the run times out. Its registers
  check fails as well.
- **tlast:** s_axis has it at the end of every 4th job and of the last one. m_axis has it at
  the end of exactly those jobs' D.
- **Cycles per job, with neither side held back:**
  - f16: 2,304 beats in + 8,192 of compute + 512 beats out + 19 of latency and control =
    11,027;
  - f32: 2,560 in + 8,192 + 1,024 out + 19 = 11,795.
- The D of the first run of each type, compared element by element, is in
  `../gpu/h029-gemm-17d8b7f/rtx2060s/compare-rtl-model-*.json`: 0 differences with the GPU
  and with the model.

`tm_pl_array-h029-17d8b7f.jsonl` holds the testbench's output lines.
