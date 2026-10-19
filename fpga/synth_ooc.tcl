# Out-of-context synthesis of the registered Volta T-FDPA unit for the KV260's XCK26.
#   vivado -mode batch -source fpga/synth_ooc.tcl -tclargs <out dir> [clock period ns] [PIPE]
set out [lindex $argv 0]
set period [expr {[llength $argv] > 1 ? [lindex $argv 1] : 4.0}]
set pipe [expr {[llength $argv] > 2 ? [lindex $argv 2] : 0}]
set here [file dirname [file normalize [info script]]]
file mkdir $out

read_verilog -sv [list $here/../rtl/tfdpa_f16.sv $here/tfdpa_f16_reg.sv]
synth_design -top tfdpa_f16_reg -part xck26-sfvc784-2LV-c -mode out_of_context -generic PIPE=$pipe
create_clock -name clk -period $period [get_ports clk]
opt_design
place_design
route_design
report_utilization -file $out/utilization.rpt
report_timing_summary -file $out/timing.rpt
report_timing -max_paths 1 -file $out/critical_path.rpt
puts "period_ns $period pipe $pipe"
puts "wns_ns [get_property SLACK [get_timing_paths -max_paths 1]]"
