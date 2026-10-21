# tm_pl on the KV260 (bitstream 3e535a9: Volta's measured e_zero)

The `3e535a9` bitstream (`../kv260-3e535a9/`) on the KV260, loaded with
`tools/kv260/load.sh 3E535A90`. The three units carry the `e_zero` measured on four V100s
(`rtl/tfdpa_f16.sv`): −131 for C in f32, −25 for C in f16 with D in f32, and −19 for C in
f16 with D in f16. MMA-Sim's −20 was used by `35ffa34`. The cases are those of
`../kv260-35ffa34-board/`, byte for byte. The expected words come from the model of
`f4fe62c` (`measured=True`).

## The clock

The design meets 200 MHz (WNS +0.274 ns) and 187.5 MHz (+0.607 ns), but not the 214.29 MHz
that the board gives for a 200 MHz request (−0.059 ns; `sta_*.txt` in `../kv260-3e535a9/`).
So the run is at 187.5 MHz, with the overlay asking for 187,500,000 Hz.

The area grew from 7,918 to 8,567 LUTs. In `35ffa34` the two f16-C units shared their front
stages: same `EZERO`, same inputs, and `u_f16` used only 129 LUTs. With different `EZERO`
they no longer share.

## Results (187,499,999 Hz)

| variant | cases | mismatches (f32 / f32c16 / f16 unit) | nonzero padding | cases/s |
|---|---|---|---|---|
| f32.f16.f16.f32 | 1,000,011 | 0 / 0 / 0 | 0 | 0.936 × 10⁸ |
| f32.f16.f16.f16 | 1,000,010 | 0 / 0 / 0 | 0 | 0.936 × 10⁸ |
| f16.f16.f16.f16 | 1,000,010 | 0 / 0 / 0 | 0 | 0.935 × 10⁸ |

**Against `35ffa34`'s board results,** on the same cases, exactly 6 words differ: per source
variant, one in the f32c16 unit and one in the f16 unit, and none in the f32 unit. These are
the directed `e_zero` cases, where `35ffa34` gave MMA-Sim's value and this build gives the
V100's.

The binary results are not versioned; their SHA-256 is in `sha256-results.txt`.
