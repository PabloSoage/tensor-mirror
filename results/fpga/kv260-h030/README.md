# KV260 systems for H-030: the 4 x 4 array with replicated counters and C/D in LUTRAM

The systems of `../kv260-h029`, rebuilt from `3bfffdd`, not dirty. The run counters `mb`,
`nbk`, `s` and `q` carry `max_fanout = 8`, and the C/D banks carry
`ram_style = "distributed"`. The generics are in each `versions.txt`.

| types | LUT | FF | BRAM36 tiles | WNS at 214.29 MHz (ns) | at 187.5 MHz | H-029 at 214.29 MHz |
|---|---|---|---|---|---|---|
| f16 | 46,920 | 34,299 | 39 | **+0.042** | +0.708 | −0.053 |
| f32 | 48,854 | 35,220 | 39 | **−0.082** | +0.584 | −0.081 |

- **f16 now closes at 214.29 MHz.** f32 does not.
- **f32's worst paths are now different**, from the routed checkpoint at 4.6667 ns:
  - the worst (−0.082 ns) runs from the DMA's MM2S output register to the write data of an
    A bank, with no logic between: s_axis_tdata reaches 32 BRAMs;
  - next come two paths inside the DMA (its stream reset into its skid buffer, −0.057 ns);
  - then one inside a unit, at three levels of logic (−0.045 ns).
- **LUT:** `tm_pl_array` takes 41,265 in f16, 2,236 more than in H-029. Of them, 2,048 are
  LUTRAM: 128 per bank, two LUT per bit for a read port beside the write port.
- **The array's bits do not change:** in Verilator both types give the D of H-029 byte for
  byte (`../../rtl/tm_pl_array-h030-3bfffdd.jsonl`).

The images are not kept here; `sha256.txt` names them.
