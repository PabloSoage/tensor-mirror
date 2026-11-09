# Vivado GUI script for the README's schematics: the elaborated RTL of one T-FDPA unit
# (Ampere f16, as in the arrays) and of a 2 x 2 tm_array, written as SVG, then exits.
#   vivado -mode gui -source fpga/render_schematic.tcl -tclargs <out dir>
lassign $argv out
file mkdir $out
set root [file normalize [file dirname [info script]]/..]
read_verilog -sv [lmap f {pipe_max pipe_sum pipe_delay tfdpa tm_array} {file join $root rtl $f.sv}]
set unit {L=8 F=24 EZERO=-20 RNE=1 AW=16 AE=5 AM=10 BW=16 BE=5 BM=10 CE=5 CM=10 DE=5 DM=10 XP=1 SP=1}
proc draw {top generics name file} {
    synth_design -rtl -top $top -part xck26-sfvc784-2LV-c -generic $generics
    show_schematic -name $name [concat [get_cells] [get_ports] [get_nets]]
    write_schematic -force -format svg -orientation landscape -scope all -name $name $file
    close_design
}
draw tfdpa [concat $unit PIPE=1] unit $out/tfdpa_unit.svg
draw tm_array [concat $unit R=2 C=2 Q=16] array $out/tm_array_2x2.svg
puts "render_schematic.tcl: done"
exit
