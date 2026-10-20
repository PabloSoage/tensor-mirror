# tm_pl on the KV260 (bitstream 35ffa34)

The `35ffa34` bitstream (`../kv260-35ffa34/`) on a KV260 (Ubuntu 22.04.5, kernel
5.15.0-1080-xilinx-zynqmp), loaded with `tools/kv260/load.sh 35FFA347`. One million
random cases per source variant plus the directed ones (`tools/pl_vectors.py`, seed 1).
Every case runs through all three units, run with `tools/kv260/tm_run.py` and checked on
the host with `tools/kv260/board_cases.py check`.

## The clock

The overlay asks pl_clk0 for 200 MHz. The board's boot firmware fixes IOPLL at
1,499,999,985 Hz, so Linux gives IOPLL/7 = **214,285,713 Hz**, and 200 MHz is not
available. The routed checkpoint was re-timed at the rates the board can give
(`fpga/sta_rates.tcl`, reports `sta_*.txt`):

| tag | pl_clk0 | period (ns) | WNS (ns) | WHS (ns) |
|---|---|---|---|---|
| base | 200 MHz (as built) | 5.000 | +0.558 | +0.010 |
| div8 | 187.5 MHz | 5.333 | +0.891 | +0.010 |
| div7 | 214.29 MHz | 4.667 | +0.225 | +0.010 |

## Results

| run | pl_clk0 (Hz) | variant | cases | mismatches (3 units) | nonzero padding | cases/s |
|---|---|---|---|---|---|---|
| 214 | 214,285,713 | f32.f16.f16.f32 | 1,000,011 | 0 / 0 / 0 | 0 | 1.065 × 10⁸ |
| 214 | 214,285,713 | f32.f16.f16.f16 | 1,000,010 | 0 / 0 / 0 | 0 | 1.065 × 10⁸ |
| 214 | 214,285,713 | f16.f16.f16.f16 | 1,000,010 | 0 / 0 / 0 | 0 | 1.066 × 10⁸ |
| 187 | 187,499,999 | the same three | the same | bit-identical to run 214 | | 0.935 × 10⁸ |

- **Run 187** used the same files with the overlay asking 187,500,000 Hz (the `.dtsi`
  with that rate, `dtc -@`). Its three result files have the SHA-256 of run 214's
  (`sha256-results.txt`).
- **Counters:** in every transfer, `CASES_IN` = `CASES_OUT` = the number of cases, with no
  DMA error.
- **Throughput:** 0.497 and 0.499 cases per cycle. That is the 128-bit HP0 read ceiling:
  one 32-byte case takes two beats.
- **The binary results are not versioned;** their SHA-256 is in `sha256-results.txt`.

Files:
- `run-214.log`, `run-187.log`: `tm_run.py`'s output;
- `check-*-214.json`: `board_cases.py check`;
- `board.txt`: the 187.5 MHz `.dtbo`'s SHA-256 and the OS.
