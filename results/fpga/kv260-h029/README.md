# KV260 system for H-029: a 4 x 4 array of Ampere f16 units

`fpga/build_kv260.tcl top=tm_pl_array` with R = C = 4, Q = 16, one job of M x N x K =
64 x 64 x 256, and the A100's `m16n8k8.f16.f16.f16.f16` unit (`EZERO:-20`, XP = SP = 1; the
generics are in `versions.txt`). Built from `17d8b7f`, not dirty. pl_clk0 was asked at
200 MHz. The board gives IOPLL / 7 = 214.29 MHz (`sta_div7.txt`) or IOPLL / 8 = 187.5 MHz
(`sta_div8.txt`).

| | LUT | FF | DSP | BRAM36 tiles | WNS at 214.29 MHz (ns) | at 187.5 MHz |
|---|---|---|---|---|---|---|
| the system, routed | 44,681 | 33,244 | 128 | 55 | **−0.053** | +0.613 |
| `tm_pl_array` | 39,029 | 23,213 | 128 | 48 | | |
| of it, the 16 units (`u_array`) | 38,120 | 22,551 | 128 | 0 | | |

- **It does not close at 214.29 MHz** by 53 ps, so the board runs it at 187.5 MHz, with the
  overlay's `assigned-clock-rates` set to 187,500,000.
- **Per unit:** 2,383 LUT, against 2,078 out of context (H-018). The difference is the
  multiplexer on c and the slots' feedback.
- **BRAM:**
  - A and B take 16 BRAM36 each: 4 banks of 256 words of 256 bits.
  - The 16 C/D banks take one BRAM36 each, although each holds only 64 words of 64 bits.
    They would fit in LUTRAM.
  - The DMA takes the other 6 BRAM36 and 2 BRAM18.

The images are not kept here; `sha256.txt` names them.
