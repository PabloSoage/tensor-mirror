# Re-time a routed checkpoint at other pl_clk0 periods, without re-implementing it.
#   vivado -mode batch -source fpga/sta_rates.tcl -tclargs <routed .dcp> <out dir> <tag> <period ns> ...
# The KV260's boot firmware fixes the PS PLLs (IOPLL 1,499,999,985 Hz), so a pl_clk0 asked
# of Linux comes out as IOPLL / an integer: 200 MHz gives /7 = 214.29 MHz, /8 is 187.5 MHz.
# Vivado's ACT_FREQMHZ uses the preset's PLLs and does not say what the board will run at.
# Each tag redefines clk_pl_0 on its source pin and writes <out dir>/sta_<tag>.txt.
lassign $argv dcp out
file mkdir $out
open_checkpoint $dcp
set c [get_clocks clk_pl_0]
set src [get_pins [get_property SOURCE_PINS $c]]
foreach {tag period} [lrange $argv 2 end] {
    create_clock -name clk_pl_0 -period $period $src
    set s [get_property SLACK [get_timing_paths -max_paths 1 -setup]]
    set h [get_property SLACK [get_timing_paths -max_paths 1 -hold]]
    puts "sta_rates.tcl: $tag period $period ns WNS $s WHS $h"
    report_timing_summary -no_header -no_detailed_paths -file $out/sta_$tag.txt
}
