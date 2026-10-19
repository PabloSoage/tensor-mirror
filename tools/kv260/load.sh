#!/bin/bash
# load.sh - put the tensor-mirror PL on the KV260 and check it answers. Run as root on the
# board, from a directory holding tensor-mirror.bit.bin (fpga/build_kv260.tcl),
# tensor-mirror.dtbo (dtc -@ -I dts -O dtb -o tensor-mirror.dtbo tensor-mirror.dtsi) and
# shell.json:
#
#   sudo bash load.sh [expected BUILD, hex: the first 8 digits of the commit] [udmabuf bytes]
#
# It installs the files as the xmutil app "tensor-mirror", unloads the app in the slot,
# loads this one, checks pl_clk0 runs (an AXI access with the PL clock off hangs the PS),
# sets HPM0 to 32 bits in FPD_SLCR.AFI_FS (the Kria boot leaves 128), checks that the AFI
# behind HP0 is out of reset, reads tm_pl's ID, VERSION and BUILD, and loads u-dma-buf
# with one buffer (udmabuf0) if it is not loaded yet. Exit 0 only if it all checks out.
set -euo pipefail
want_build=${1:-}
buf_bytes=${2:-67108864}
app=/lib/firmware/xilinx/tensor-mirror
mkdir -p "$app"
cp tensor-mirror.bit.bin tensor-mirror.dtbo shell.json "$app/"
xmutil unloadapp >/dev/null 2>&1 || true
xmutil loadapp tensor-mirror
[ "$(cat /sys/class/fpga_manager/fpga0/state)" = operating ] || { echo "load.sh: fpga0 not operating" >&2; exit 1; }
mountpoint -q /sys/kernel/debug || mount -t debugfs none /sys/kernel/debug
echo "pl0_ref $(cat /sys/kernel/debug/clk/pl0_ref/clk_rate) Hz, enable count $(cat /sys/kernel/debug/clk/pl0_ref/clk_enable_count)"
[ "$(cat /sys/kernel/debug/clk/pl0_ref/clk_enable_count)" -gt 0 ] || { echo "load.sh: pl_clk0 is off" >&2; exit 1; }
afi=$(busybox devmem 0xFD615000 32)
busybox devmem 0xFD615000 32 "$(printf '0x%08X' $(( afi & ~0x300 )))"
echo "AFI_FS $afi -> $(busybox devmem 0xFD615000 32) (HPM0 at 32 bits)"
rst=$(busybox devmem 0xFD1A0100 32)
echo "RST_FPD_TOP $rst"
[ $(( rst & 0x200 )) -eq 0 ] || { echo "load.sh: AFI_FM2 (HP0) is in reset" >&2; exit 1; }
id=$(busybox devmem 0xA0000000 32); ver=$(busybox devmem 0xA0000004 32); build=$(busybox devmem 0xA0000008 32)
echo "ID $id, VERSION $ver, BUILD $build, LATENCY $(busybox devmem 0xA000000C 32), F $(busybox devmem 0xA0000010 32)"
[ $((id)) -eq $((0x544D4952)) ] && [ $((ver)) -eq 1 ] || { echo "load.sh: wrong ID or VERSION" >&2; exit 1; }
if [ -n "$want_build" ] && [ $((build)) -ne $((16#$want_build)) ]; then
    echo "load.sh: BUILD is not $want_build" >&2; exit 1
fi
if [ ! -e /dev/udmabuf0 ]; then
    ko=$(find / -xdev -name u-dma-buf.ko 2>/dev/null | head -1)
    [ -n "$ko" ] || { echo "load.sh: no u-dma-buf.ko on this board" >&2; exit 1; }
    insmod "$ko" udmabuf0="$buf_bytes"
fi
echo "udmabuf0 $(cat /sys/class/u-dma-buf/udmabuf0/size) bytes at $(cat /sys/class/u-dma-buf/udmabuf0/phys_addr)"
echo "load.sh: tensor-mirror answers"
