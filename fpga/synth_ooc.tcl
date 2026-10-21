# Out-of-context synthesis of the registered Volta T-FDPA unit for the KV260's XCK26.
#   vivado -mode batch -source fpga/synth_ooc.tcl -tclargs <out dir> [clock period ns] [PIPE] [C_F16] [D_F16]
set out [lindex $argv 0]
set period [expr {[llength $argv] > 1 ? [lindex $argv 1] : 4.0}]
set pipe [expr {[llength $argv] > 2 ? [lindex $argv 2] : 0}]
set cf16 [expr {[llength $argv] > 3 ? [lindex $argv 3] : 0}]
set df16 [expr {[llength $argv] > 4 ? [lindex $argv 4] : 0}]
set here [file dirname [file normalize [info script]]]
file mkdir $out

read_verilog -sv [list $here/../rtl/pipe_max.sv $here/../rtl/pipe_sum.sv $here/../rtl/pipe_delay.sv \
    $here/../rtl/tfdpa.sv $here/../rtl/tfdpa_f16.sv $here/tfdpa_f16_reg.sv]
synth_design -top tfdpa_f16_reg -part xck26-sfvc784-2LV-c -mode out_of_context \
    -generic PIPE=$pipe -generic C_F16=$cf16 -generic D_F16=$df16
create_clock -name clk -period $period [get_ports clk]
opt_design
place_design
route_design
report_utilization -file $out/utilization.rpt
report_timing_summary -file $out/timing.rpt
report_timing -max_paths 1 -file $out/critical_path.rpt
puts "period_ns $period pipe $pipe c_f16 $cf16 d_f16 $df16"
puts "wns_ns [get_property SLACK [get_timing_paths -max_paths 1]]"
