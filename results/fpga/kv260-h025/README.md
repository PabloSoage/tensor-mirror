# KV260 system for H-025: A100 m16n8k8.f16.f16.f16.f16 with e_zero = -20

As `../kv260-h020/a100-m16n8k8.f16.f16.f16.f16` (bitstream a3), with `EZERO:-20` instead of
-21 after the model's change (`60213eb`, not dirty; `versions.txt`). pl_clk0 asked at 200 MHz;
on the board IOPLL / 7 = 214.29 MHz (`sta_div7.txt`), `sta_div8.txt` is 187.5 MHz.

| LUT | FF | DSP | WNS at 214.29 MHz (ns) | at 187.5 MHz |
|---|---|---|---|---|
| 7,672 | 11,606 | 8 | +0.504 | +1.170 |

The H-020 build with -21 had 7,684 LUT and +0.312 ns. The images are not kept here;
`sha256.txt` names them.
