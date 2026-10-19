# tensor-mirror

Bit-exact replicas of GPU tensor-core arithmetic, in hardware.

GPU matrix units do not compute `D = A·B + C` the IEEE way: they align the products at the
largest exponent, truncate them to a fixed number of fractional bits, sum in fixed point and
round once, with parameters that change from one generation to the next. Bit-accurate
software models of that behaviour now exist, notably
[MMA-Sim](https://github.com/microsoft/MMA-Sim) ([arXiv 2511.10909](https://arxiv.org/abs/2511.10909)).
tensor-mirror takes them to RTL: a parametrisable MMA unit for an FPGA (AMD Kria KV260) that
returns the same bits as a chosen NVIDIA GPU.

Status: early. What exists today:

| Path | What |
|---|---|
| `python/tensormirror/formats.py` | Floating-point formats as integer fields; RZ and RNE rounding |
| `python/tensormirror/tfdpa.py` | Truncated fused dot-product-add (T-FDPA) in integer arithmetic, written the way the datapath will be |
| `python/tensormirror/arch.py` | Published T-FDPA parameters per architecture (`L_max`, `F`, rounding, zero exponent) |
| `tools/xcheck_mmasim.py` | Random bit-pattern comparison against MMA-Sim, with parameter mutants |

Planned: own `mma.sync` kernels to check both models against real GPUs (V100 first), then
the Volta unit in SystemVerilog, then the parametrised unit (Ampere to Blackwell, including
MXFP4/NVFP4 block scaling), on the board.

## Running

```sh
python -m venv .venv && .venv/bin/pip install -e .[test]
.venv/bin/pytest
```

The MMA-Sim comparison needs `pip install -e .[mmasim]` on Linux (MMA-Sim builds a C++
extension at import time, with `ninja` on the PATH):

```sh
python tools/xcheck_mmasim.py --arch Volta --types f32.f16.f16.f32 --cases 1000000
```

## Licence

Apache License 2.0, see [`LICENSE`](LICENSE). MMA-Sim, used as a test dependency, is MIT.
