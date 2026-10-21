# KV260 system with tm_pl_x as mxf8f6f4 E4M3 x E4M3 (H-019)

`fpga/build_kv260.tcl` at `1bbf53c`: `top tm_pl_x` with the generics of
`m16n8k32.mxf8f6f4.1x.f32.e4m3.e4m3.f32.ue8m0` (`tools/rtl_params.py bs`) and `XP:1 SP:1`
(`versions.txt`). pl_clk0 asked at 200 MHz; on the board that is IOPLL / 7 = 214.29 MHz
(`fpga/sta_rates.tcl`: `sta_div7.txt` at 4.6667 ns, WNS +0.196 ns; `sta_div8.txt` at
187.5 MHz, WNS +0.862 ns; at Vivado's 200 MHz, WNS +0.529 ns). The images are not kept
here; `sha256.txt` names them.
