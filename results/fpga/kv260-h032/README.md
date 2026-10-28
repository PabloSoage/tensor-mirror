# KV260 systems for H-032: the 4 x 4 array with a registered reset

The systems of `../kv260-h031`, rebuilt from `5e29cd9`, not dirty. `tm_pl_array` registers
`pl_resetn` twice, the second copy replicated (`max_fanout = 16`). It and `tm_regs` use that
copy, and s_axis is not ready while it holds.

| types | LUT | FF | BRAM36 tiles | WNS at 214.29 MHz (ns) | at 187.5 MHz | H-031 at 214.29 MHz |
|---|---|---|---|---|---|---|
| f16 | 47,355 | 36,688 | 39 | **+0.108** | +0.774 | +0.027 |
| f32 | 48,858 | 37,568 | 39 | **+0.027** | +0.693 | −0.059 |

- **Both types close at 214.29 MHz.** f32's worst paths now run from the registered beat's bank
  address to an A bank's write address, with no logic between, at +27 ps
  (`f32.f16.f16.f32/worst_paths_div7.txt`).
- **The bits do not change:** Verilator gives the D of H-029 in both types, with 11,028 and
  11,796 cycles per job, as in H-031 (`../../rtl/tm_pl_array-h032-5e29cd9.jsonl`).
- **On the board at 214.29 MHz** both give the GPUs' D byte for byte
  (`../kv260-board-2026-10-28/c16`, `c32`).

The images are not kept here; `sha256.txt` names them.
