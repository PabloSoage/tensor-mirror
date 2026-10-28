# tensor-mirror

Bit-exact replicas of GPU tensor-core arithmetic, in hardware.

GPU matrix units do not compute `D = A·B + C` the IEEE way. They align the products at the
largest exponent, truncate them to a fixed number of fractional bits, sum in fixed point and
round once, with parameters that change from one generation to the next. Bit-accurate
software models of that behaviour exist, notably [MMA-Sim](https://github.com/microsoft/MMA-Sim)
([arXiv 2511.10909](https://arxiv.org/abs/2511.10909)). tensor-mirror takes them to RTL: one
parametrisable unit for an FPGA (AMD Kria KV260) that returns the same bits as a chosen
NVIDIA GPU, from Volta to RTX Blackwell, including MXFP4/NVFP4 block scaling.

## Status

- **Integer model** (`python/tensormirror/`): T-FDPA, ST-FDPA and GST-FDPA (block scaling)
  with integer arithmetic only, written the way the datapath is built.
  - `measured=False` gives the published parameters. It equals MMA-Sim on random bit
    patterns in every T-FDPA `mma.sync` MMA-Sim models: 130 instructions in v2.1.1
    (`results/xcheck/`).
  - `measured=True` (the default) applies what we measured on real GPUs.
- **Checked against real GPUs** with our own `mma.sync` kernels (`gpu/`, `results/gpu/`):
  - **Volta:** four V100;
  - **Turing:** an RTX 2060 and an RTX 2060 Super;
  - **Ampere:** an A100;
  - **RTX Blackwell:** an RTX 5070 Ti, with `kind::f8f6f4` and the `block_scale` forms.
  - Every random and directed case matches the measured model.
- **What we measured that differs from the published models**:
  - **Turing:** it runs Volta's `m8n8k4` with Volta's 23-bit sum (F = 23), not its own 24.
  - **`e_zero`, the exponent a zero accumulator takes, depends on D's type:**
    - Volta: −19 with D in f16;
    - Turing and Ampere: −20 with D in f16 (it was −21).
  - **`kind::f8f6f4` reads an E2M1 byte as E2M3:** bits 1:0 are extra mantissa bits, not
    padding.
  - MMA-Sim adopted the first two in v2.1.0 and v2.1.1.
- **RTL** (`rtl/`, verified with Verilator against the model, with mutants):
  - **`tfdpa.sv`:** generic T-FDPA, with an optional per-block scale (ST-FDPA);
  - **`gst_fdpa.sv`:** MXFP4/NVFP4 group scaling;
  - both are pipelined and meet 250 MHz out of context in every configuration from Volta to
    RTX Blackwell (`results/fpga/ooc-*`).
- **On the KV260** (`rtl/tm_pl_x.v`, `fpga/build_kv260.tcl`):
  - **what runs:** one unit behind an AXI DMA at 214.29 MHz;
  - **result:** 0 mismatches with the GPUs in 7.6 million results of 13 instructions
    (`results/fpga/kv260-board-2026-10-22/`):
    - A100 f16, bf16 and TF32;
    - RTX 5070 Ti FP8, MXFP8 E4M3, MXFP4 and NVFP4;
    - instructions whose K spans several groups, computed in two passes.

Next: an array of units with on-chip operand buffers. One unit already waits on the DMA's
input, so more units only pay off with operand reuse.

## Layout

| Path | What |
|---|---|
| `python/tensormirror/` | Formats, T-FDPA, block scaling, parameters per architecture |
| `gpu/` | `mma.sync` harnesses on raw bits (one instruction, or a loop of them per tile), block scale, throughput |
| `tools/` | Case generators and checks (`gpu_cases.py`, `bs_cases.py`), probes, MMA-Sim cross-checks, RTL vector writers, run scripts per experiment |
| `rtl/`, `tb/` | SystemVerilog/Verilog units and the PL top; Verilator testbenches |
| `fpga/` | Out-of-context synthesis, the KV260 system build, timing at the board's clock rates |
| `results/` | Evidence for every number: GPU runs, cross-checks, RTL runs, synthesis and board runs, each with a README |

## Running

```sh
python -m venv .venv && .venv/bin/pip install -e .[test]
.venv/bin/pytest
```

The MMA-Sim comparison needs `pip install -e .[mmasim]` on Linux. MMA-Sim builds a C++
extension at import time, with `ninja` on the PATH.

```sh
python tools/xcheck_mmasim.py --arch Volta --inst m8n8k4.f32.f16.f16.f32 --cases 1000000
python tools/xcheck_mmasim.py --all --measured     # the measured model, every instruction
```

GPU runs: `tools/gpu_cases.py gen`, then the harness on the GPU host
(`tools/gpu_batch.sh run`), then `tools/gpu_cases.py check`. RTL: Verilator 5
(`tools/run_rtl_generic.sh`, `tools/run_pl_x.sh`). FPGA: Vivado 2026.1.

## Related work

- **MMA-Sim:** bit-accurate software models of 10 architectures. It is our reference.
- **Khattak and Mikaitis** ([arXiv 2512.07004](https://arxiv.org/abs/2512.07004)): models of
  V100, A100, H100 and B200.
- **[ucb-bar/nvfp4](https://github.com/ucb-bar/nvfp4):** an NVFP4 tensor core on an
  Artix-7, compared with an RTX 5090.

## Licence

Apache License 2.0, see [`LICENSE`](LICENSE). MMA-Sim, used as a test dependency, is MIT.
