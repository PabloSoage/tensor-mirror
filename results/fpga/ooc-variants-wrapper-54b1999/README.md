# Volta's three variants with tfdpa_f16 as a wrapper of tfdpa, out of context (T-3.4)

`fpga/synth_ooc.tcl` (PIPE=1, 4.0 ns), the flow of `ooc-variants-f1cbafc` (H-013), with
`rtl/tfdpa_f16.sv` instantiating `rtl/tfdpa.sv` (since `3541806`; `tfdpa.sv` as at `e917de5`,
XP = SP = 0 here). The run started at `54b1999`; the files it reads are those of `e917de5`.

| variant (C, D) | LUT | FF | DSP | WNS (ns) | H-013: LUT, FF, WNS |
|---|---|---|---|---|---|
| f32, f32 | 1,207 | 815 | 4 | +0.093 | 1,149, 795, +0.036 |
| f16, f32 | 1,161 | 741 | 4 | +0.379 | 1,070, 699, +0.568 |
| f16, f16 | 1,210 | 761 | 4 | +0.282 | 1,222, 751, +0.280 |
