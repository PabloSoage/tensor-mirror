#!/usr/bin/env bash
# Copy the small evidence of a gpu_batch.sh or bs_batch.sh root into results/gpu/<name>-<commit>/: the
# summary, the GPU description, every check.json (or probe.json) and run.json, and the SHA-256 of every
# input and output file (the files themselves are regenerated from the seed and the GPU).
#   tools/keep_gpu_results.sh <root> <name>
set -eu
root=$1
dest=results/gpu/$2-$(git rev-parse --short HEAD)
mkdir -p "$dest"
cp "$root/summary.jsonl" "$root/gpu.csv" "$dest/"
for dir in "$root"/*/; do
    inst=$(basename "$dir")
    mkdir -p "$dest/$inst"
    cp "$dir/run.json" "$dir/cases.json" "$dest/$inst/"
    for f in check.json probe.json; do [ -f "$dir/$f" ] && cp "$dir/$f" "$dest/$inst/"; done
    (cd "$dir" && sha256sum *.bin) > "$dest/$inst/sha256.txt"
done
echo "$dest"
