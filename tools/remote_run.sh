#!/usr/bin/env bash
# On the GPU host's WSL, from the directory that holds <archive>.tar and gpu_batch.sh:
# unpack into ~/tensor-mirror-run, run the harness on every case directory, and pack the
# outputs (d.bin, run.json, gpu.csv) into <archive>-out.tar next to the input archive.
#   bash remote_run.sh <archive name without .tar> <case root inside it> <binary inside it>
set -eu
name=$1 root=$2 bin=$3
src=$PWD
work=~/tensor-mirror-run
mkdir -p "$work"
cd "$work"
rm -rf "$root" "$bin"
tar -xf "$src/$name.tar"
cp "$src/gpu_batch.sh" .
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader
time bash gpu_batch.sh run "$root" "./$bin"
ls "$root"/*/d.bin | wc -l
tar -cf "$src/$name-out.tar" "$root"/gpu.csv "$root"/*/d.bin "$root"/*/run.json
