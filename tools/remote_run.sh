#!/usr/bin/env bash
# On the GPU host's WSL, from the directory that holds <archive>.tar and gpu_batch.sh:
# unpack into ~/tensor-mirror-run, run the harness on every case directory, and pack the
# outputs (d.bin, run.json, gpu.csv) into <archive>-out.tar next to the input archive.
#   bash remote_run.sh <archive name without .tar> <case root inside it> <binary inside it> [batch script]
# The batch script (default gpu_batch.sh) must sit next to the archive.
set -eu
name=$1 root=$2 bin=$3 batch=${4:-gpu_batch.sh}
src=$PWD
work=~/tensor-mirror-run
mkdir -p "$work"
cd "$work"
rm -rf "$root" "$bin"
tar -xf "$src/$name.tar"
cp "$src/$batch" .
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader
time bash "$batch" run "$root" "./$bin"
ls "$root"/*/d.bin | wc -l
tar -cf "$src/$name-out.tar" "$root"/gpu.csv "$root"/*/d.bin "$root"/*/run.json
