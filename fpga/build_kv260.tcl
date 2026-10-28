# build_kv260.tcl - the KV260 system around rtl/tm_pl.v, rtl/tm_pl_x.v or rtl/tm_pl_array.v,
# from nothing, in batch mode:
# block design, synthesis, implementation, bitstream and the .bit.bin for the board's
# FPGA manager. Tcl only, for Vivado 2026.1 on Windows.
#
#   vivado.bat -mode batch -nojournal -nolog -source fpga/build_kv260.tcl -tclargs key=value ...
#
# Keys [defaults]:
#   work=<dir>       scratch directory [<repo>/build/kv260]; on Windows give a short one,
#                    or the IP paths pass MAX_PATH
#   out=<dir>        where the evidence and the images go [<work>/out]
#   pl_mhz=<MHz>     pl_clk0 [200]; the PLL that comes closest is used, and what it
#                    gives goes to <out>/pl_clk.txt. fpga/kv260/tensor-mirror.dtsi sets
#                    the rate on the board and must match. On the KV260 the board
#                    gives IOPLL (1.5 GHz) / an integer, not what Vivado reports
#                    (fpga/sta_rates.tcl re-times the routed design at that rate)
#   stop=synth|route|bitstream [bitstream]
#   jobs=<n>         [6]
#   top=tm_pl|tm_pl_x|tm_pl_array [tm_pl]
#   NAME:value ...   the top's parameters (tfdpa's, from tools/rtl_params.py, with XP and
#                    SP), one argument each: cmd splits arguments at '=' and at ','
#
# The system:
#   ps     Zynq UltraScale+ MPSoC with the KV260 preset; pl_clk0 clocks the whole PL;
#          HPM0 (FPD) at 32 bits for AXI4-Lite, HP0 (FPD) at 128 bits for the DMA,
#          pl_ps_irq0 for its interrupts
#   tm     the top as a module reference, BUILD = the first 8 hex digits of the
#          commit, and the generics
#   dma    AXI DMA in simple (register) mode: MM2S 256-bit stream into tm's s_axis,
#          S2MM 128-bit stream from tm's m_axis, 40-bit addresses, 26-bit lengths
#   hp0    SmartConnect from the DMA's MM2S and S2MM masters to HP0
#   lite   SmartConnect from HPM0 to tm's registers (4 KB at 0xA000_0000) and the
#          DMA's (64 KB at 0xA001_0000)
#   rst    proc_sys_reset of pl_clk0 from pl_resetn0
#
# Writes to <out>: versions.txt, pl_clk.txt, utilisation and timing reports after
# synthesis and routing, summary_<stage>.txt, and with stop=bitstream tensor-mirror.bit,
# tensor-mirror.bit.bin (bootgen) and tensor-mirror.xsa.

set root [file normalize [file join [file dirname [info script]] ..]]
set part xck26-sfvc784-2LV-c
set board xilinx.com:kv260_som:part0:2.0

array set opt {pl_mhz 200 stop bitstream jobs 6 top tm_pl generics {}}
set pairs {}
foreach a $argv {  ;# cmd splits key=value at '=': both forms give the same pairs
    if {[regexp {^[A-Z][A-Z0-9_]*:-?[0-9]+$} $a]} {
        lappend opt(generics) $a
    } elseif {[regexp {^([a-z_]+)=(.*)$} $a -> k v]} {
        lappend pairs $k $v
    } else {
        lappend pairs $a
    }
}
foreach {k v} $pairs { set opt($k) $v }
if {![info exists opt(work)]} { set opt(work) $root/build/kv260 }
if {![info exists opt(out)]} { set opt(out) $opt(work)/out }
if {[lsearch -exact {synth route bitstream} $opt(stop)] < 0} { error "stop=$opt(stop)" }
if {[lsearch -exact {tm_pl tm_pl_x tm_pl_array} $opt(top)] < 0} { error "top=$opt(top)" }

# ---- provenance, before anything is written ----
set rev [string trim [exec git -C $root rev-parse HEAD]]
set dirty [expr {[string length [string trim [exec git -C $root status --porcelain --untracked-files=no]]] > 0}]
set build_hash [scan [string range $rev 0 7] %x]
file mkdir $opt(work) $opt(out)
set fh [open [file join $opt(out) versions.txt] w]
puts $fh "vivado [version -short]"
puts $fh "tensor-mirror $rev[expr {$dirty ? " (dirty: rebuild after commit)" : ""}]"
puts $fh "part $part"
puts $fh "pl_mhz $opt(pl_mhz)"
puts $fh "top $opt(top)"
puts $fh "generics $opt(generics)"
puts $fh "date [clock format [clock seconds] -format %Y-%m-%dT%H:%M:%S]"
close $fh

proc ipdef {name} {
    set l [lsort -dictionary [get_ipdefs -all xilinx.com:ip:${name}:*]]
    if {![llength $l]} { error "no IP $name in this Vivado" }
    return [lindex $l end]
}

proc reports {out stage} {
    report_utilization -file [file join $out utilization_$stage.rpt]
    report_utilization -hierarchical -hierarchical_depth 4 -file [file join $out utilization_hier_$stage.rpt]
    report_timing_summary -max_paths 20 -report_unconstrained -file [file join $out timing_summary_$stage.rpt]
    set util [report_utilization -return_string]
    set tim [report_timing_summary -no_header -no_detailed_paths -return_string]
    set lines [list "stage $stage"]
    foreach {key pat} {
        lut {\|\s*CLB LUTs\*?\s*\|\s*(\d+)}
        ff {\|\s*CLB Registers\s*\|\s*(\d+)}
        dsp {\|\s*DSPs\s*\|\s*(\d+)}
        bram_tile {\|\s*Block RAM Tile\s*\|\s*([\d.]+)}
    } {
        lappend lines "$key [expr {[regexp $pat $util -> n] ? $n : "?"}]"
    }
    if {[regexp {WNS\(ns\).*?\n[-\s]+\n\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(\d+)\s+(\d+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(\d+)\s+(\d+)} \
            $tim -> wns tns tnf tne whs ths thf the]} {
        lappend lines "wns_ns $wns" "tns_ns $tns" "setup_failing $tnf" "whs_ns $whs" "ths_ns $ths" "hold_failing $thf"
    } else {
        lappend lines "wns_ns ?"
    }
    set fh [open [file join $out summary_$stage.txt] w]
    puts $fh [join $lines \n]
    close $fh
    puts "build_kv260.tcl: [join $lines {, }]"
}

# ---- project and sources ----
create_project -force tensor-mirror $opt(work)/prj -part $part
set_property board_part $board [current_project]
set_property target_language Verilog [current_project]
add_files -norecurse [list [file join $root rtl tm_regs.v] [file join $root rtl tm_pl.v] \
    [file join $root rtl tm_pl_x.v] [file join $root rtl tm_pl_array.v] [file join $root rtl tm_array.sv] \
    [file join $root rtl tfdpa_f16.sv] [file join $root rtl tfdpa.sv] \
    [file join $root rtl gst_fdpa.sv] \
    [file join $root rtl pipe_max.sv] [file join $root rtl pipe_sum.sv] [file join $root rtl pipe_delay.sv]]
update_compile_order -fileset sources_1

# ---- block design ----
create_bd_design sys
set ps [create_bd_cell -type ip -vlnv [ipdef zynq_ultra_ps_e] ps]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $ps
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} CONFIG.PSU__MAXIGP0__DATA_WIDTH {32} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} CONFIG.PSU__USE__IRQ0 {1}] $ps
set best {}
# not DPLL: it clocks the DDR, and taking it re-derives the DDR timing (CWL) into an invalid one
foreach src {IOPLL RPLL} {
    if {[catch {set_property -dict [list CONFIG.PSU__CRL_APB__PL0_REF_CTRL__SRCSEL $src \
            CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $opt(pl_mhz)] $ps}]} { continue }
    set a [get_property CONFIG.PSU__CRL_APB__PL0_REF_CTRL__ACT_FREQMHZ $ps]
    if {![llength $best] || abs($a - $opt(pl_mhz)) < abs([lindex $best 1] - $opt(pl_mhz))} { set best [list $src $a] }
}
set_property -dict [list CONFIG.PSU__CRL_APB__PL0_REF_CTRL__SRCSEL [lindex $best 0] \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $opt(pl_mhz)] $ps
set act [get_property CONFIG.PSU__CRL_APB__PL0_REF_CTRL__ACT_FREQMHZ $ps]
set fh [open [file join $opt(out) pl_clk.txt] w]
puts $fh "asked_mhz $opt(pl_mhz)\nsource [lindex $best 0]\nactual_mhz $act"
close $fh
puts "build_kv260.tcl: pl_clk0 asked $opt(pl_mhz) MHz, [lindex $best 0] gives $act MHz"

set tm [create_bd_cell -type module -reference $opt(top) tm]
set_property CONFIG.BUILD [format 0x%08X $build_hash] $tm
foreach g $opt(generics) {
    lassign [split $g :] k v
    set_property -dict [list CONFIG.$k $v] $tm  ;# -dict: a value like -133 is not an option
}

set clk [get_bd_pins ps/pl_clk0]
create_bd_cell -type ip -vlnv [ipdef proc_sys_reset] rst
connect_bd_net $clk [get_bd_pins rst/slowest_sync_clk]
connect_bd_net [get_bd_pins ps/pl_resetn0] [get_bd_pins rst/ext_reset_in]
set aresetn [get_bd_pins rst/peripheral_aresetn]
connect_bd_net $clk [get_bd_pins ps/maxihpm0_fpd_aclk] [get_bd_pins ps/saxihp0_fpd_aclk] [get_bd_pins tm/clk]
connect_bd_net $aresetn [get_bd_pins tm/pl_resetn]

set dma [create_bd_cell -type ip -vlnv [ipdef axi_dma] dma]
set_property -dict [list CONFIG.c_include_sg {0} CONFIG.c_sg_length_width {26} CONFIG.c_addr_width {40} \
    CONFIG.c_include_mm2s {1} CONFIG.c_m_axi_mm2s_data_width {256} CONFIG.c_m_axis_mm2s_tdata_width {256} \
    CONFIG.c_mm2s_burst_size {16} \
    CONFIG.c_include_s2mm {1} CONFIG.c_m_axi_s2mm_data_width {128} CONFIG.c_s_axis_s2mm_tdata_width {128} \
    CONFIG.c_s2mm_burst_size {16}] $dma
connect_bd_intf_net [get_bd_intf_pins dma/M_AXIS_MM2S] [get_bd_intf_pins tm/s_axis]
connect_bd_intf_net [get_bd_intf_pins tm/m_axis] [get_bd_intf_pins dma/S_AXIS_S2MM]

create_bd_cell -type ip -vlnv [ipdef smartconnect] hp0
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] [get_bd_cells hp0]
connect_bd_intf_net [get_bd_intf_pins dma/M_AXI_MM2S] [get_bd_intf_pins hp0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins dma/M_AXI_S2MM] [get_bd_intf_pins hp0/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins hp0/M00_AXI] [get_bd_intf_pins ps/S_AXI_HP0_FPD]

create_bd_cell -type ip -vlnv [ipdef smartconnect] lite
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2}] [get_bd_cells lite]
connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_FPD] [get_bd_intf_pins lite/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins lite/M00_AXI] [get_bd_intf_pins tm/s_axi]
connect_bd_intf_net [get_bd_intf_pins lite/M01_AXI] [get_bd_intf_pins dma/S_AXI_LITE]

foreach c {hp0 lite} {
    connect_bd_net $clk [get_bd_pins $c/aclk]
    connect_bd_net $aresetn [get_bd_pins $c/aresetn]
}
foreach pin [get_bd_pins dma/*_aclk] { connect_bd_net $clk $pin }
connect_bd_net $aresetn [get_bd_pins dma/axi_resetn]
create_bd_cell -type ip -vlnv [ipdef xlconcat] irq
set_property CONFIG.NUM_PORTS {2} [get_bd_cells irq]
connect_bd_net [get_bd_pins dma/mm2s_introut] [get_bd_pins irq/In0]
connect_bd_net [get_bd_pins dma/s2mm_introut] [get_bd_pins irq/In1]
connect_bd_net [get_bd_pins irq/dout] [get_bd_pins ps/pl_ps_irq0]

assign_bd_address
proc seg_of {name} {
    foreach seg [get_bd_addr_segs ps/Data/*] {
        if {[string match "/$name/*" [get_bd_addr_segs -of_objects $seg]]} { return $seg }
    }
    error "no address segment of $name"
}
set seg [seg_of tm]
set_property range 4K $seg
set_property offset 0xA0000000 $seg
set seg [seg_of dma]
set_property range 64K $seg
set_property offset 0xA0010000 $seg
validate_bd_design
save_bd_design
set bd [get_files sys.bd]
generate_target all $bd
add_files -norecurse [make_wrapper -files $bd -top]
set_property top sys_wrapper [current_fileset]
update_compile_order -fileset sources_1

# ---- runs ----
launch_runs synth_1 -jobs $opt(jobs)
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} { error "build_kv260.tcl: synthesis failed" }
open_run synth_1
reports $opt(out) synth
if {$opt(stop) ne "synth"} {
    set to [expr {$opt(stop) eq "bitstream" ? "write_bitstream" : "route_design"}]
    # the default run missed 200 MHz by 68 ps on the first 8-stage build; physical
    # optimisation after routing as well
    set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
    set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
    launch_runs impl_1 -to_step $to -jobs $opt(jobs)
    wait_on_run impl_1
    if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} { error "build_kv260.tcl: implementation failed" }
    open_run impl_1
    reports $opt(out) route
    report_drc -file [file join $opt(out) drc_route.rpt]
    report_methodology -file [file join $opt(out) methodology_route.rpt]
}
if {$opt(stop) eq "bitstream"} {
    set bit [glob -directory [get_property DIRECTORY [get_runs impl_1]] *.bit]
    file copy -force $bit [file join $opt(out) tensor-mirror.bit]
    write_hw_platform -fixed -include_bit -force [file join $opt(out) tensor-mirror.xsa]
    set bif [file join $opt(out) tensor-mirror.bif]
    set fh [open $bif w]
    puts $fh "all:\n\{\n  \[destination_device = pl\] [file join $opt(out) tensor-mirror.bit]\n\}"
    close $fh
    set bootgen [file join [file dirname [info nameofexecutable]] bootgen.bat]
    if {![file exists $bootgen]} { set bootgen [file join $::env(XILINX_VIVADO) bin bootgen.bat] }
    exec $bootgen -image $bif -arch zynqmp -process_bitstream bin -w on
    puts "build_kv260.tcl: [glob -directory $opt(out) *.bit.bin]"
}
exit 0
