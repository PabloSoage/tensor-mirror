#!/usr/bin/env bash
# Build gpu/tc_throughput.cu and run it on every visible GPU, logging clocks and power.
#   tools/run_tc_throughput.sh <out dir>
# Run from WSL or Linux with nvcc on PATH or in /usr/local/cuda. nvidia-smi logs at 500 ms.
set -eu
out=$1
here=$(cd "$(dirname "$0")/.." && pwd)
nvcc=$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)
mkdir -p "$out"
"$nvcc" -O3 -std=c++17 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_120,code=sm_120 "$here/gpu/tc_throughput.cu" -lcublas -o "$out/tc_throughput"
nvidia-smi --query-gpu=index,name,driver_version,vbios_version,power.limit --format=csv > "$out/gpu.csv"
"$nvcc" --version | tail -1 > "$out/host.txt"
git -C "$here" rev-parse HEAD >> "$out/host.txt"
nvidia-smi --query-gpu=timestamp,index,clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active \
    --format=csv -lms 500 > "$out/smi.csv" &
smi=$!
n=$(nvidia-smi -L | wc -l)
for ((d = 0; d < n; d++)); do "$out/tc_throughput" "$d" | tee "$out/dev$d.txt"; done
kill $smi
rm -f "$out/tc_throughput"
