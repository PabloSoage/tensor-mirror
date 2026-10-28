# KV260 systems for H-031: the 4 x 4 array with the input beat registered

The systems of `../kv260-h030`, rebuilt from `32ad55d`, not dirty. Each accepted s_axis beat
is now registered (data replicated with `max_fanout = 8`, bank, address and region) and
written to the banks one cycle later. The array starts after the job's last beat is written.

| types | LUT | FF | BRAM36 tiles | WNS at 214.29 MHz (ns) | at 187.5 MHz | H-030 at 214.29 MHz |
|---|---|---|---|---|---|---|
| f16 | 46,896 | 36,626 | 39 | **+0.027** | +0.693 | +0.042 |
| f32 | 48,816 | 37,553 | 39 | **−0.059** | +0.607 | −0.082 |

- **f32 still does not close**, but the worst path has moved again.
  `f32.f16.f16.f32/worst_paths_div7.txt` lists the 10 worst at 4.6667 ns:
  - nine run from the system reset (`proc_sys_reset`'s output flip-flop) to the reset pins of
    `tm_regs`' read-data register, with no logic between;
  - the tenth is the same reset into a counter of `tm_pl_array`.
  - Neither the banks nor the DMA are among them.
- **The bits do not change:** in Verilator both types give the D of H-029 byte for byte. Each
  job takes one cycle more, 11,028 (f16) and 11,796 (f32) with no gaps
  (`../../rtl/tm_pl_array-h031-32ad55d.jsonl`).

The images are not kept here; `sha256.txt` names them.
