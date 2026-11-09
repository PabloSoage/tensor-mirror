# Vivado GUI script for the README's pictures of a built KV260 system (fpga/build_kv260.tcl):
# the block design as SVG, then the routed design's device view with every array unit
# highlighted in its own colour, left open for a screenshot (tools/capture_vivado.ps1).
#   vivado -mode gui -source fpga/render_device.tcl -tclargs <project.xpr> <out dir>
lassign $argv xpr out
file mkdir $out
open_project $xpr
open_bd_design [get_files sys.bd]
write_bd_layout -force -format svg -orientation landscape $out/block_design.svg
close_bd_design [current_bd_design]

open_run impl_1 -name impl_1
set palette {
    {230 25 75} {60 180 75} {255 225 25} {0 130 200} {245 130 48} {145 30 180} {70 240 240}
    {240 50 230} {210 245 60} {250 190 212} {0 128 128} {220 190 255} {170 110 40}
    {255 250 200} {128 0 0} {170 255 195} {128 128 0} {255 215 180} {0 0 128} {128 128 128}
}
set units [get_cells -hier -filter {REF_NAME =~ *fdpa* && IS_PRIMITIVE == 0}]  ;# tfdpa and gst_fdpa
set i 0
foreach u $units {
    current_instance $u
    set leaves [get_cells -quiet -hier -filter {IS_PRIMITIVE}]
    current_instance
    highlight_objects -rgb [lindex $palette [expr {$i % [llength $palette]}]] $leaves
    incr i
}
puts "render_device.tcl: $i units highlighted"
set fh [open $out/ready w]
puts $fh $i
close $fh
