#!/usr/bin/env bash
# Turing's m8n8k4 on a local RTX 20xx (WSL or Linux with CUDA 12.x): the three variants and the
# two e_zero probes of the V100 (tools/probe_ezero.py, tools/probe_f16_round.py), generated,
# run and checked in <root>. Run from the repository with the tensor-mirror venv active.
#   tools/run_turing_m8n8k4.sh <root> [tiles]
set -eu
root=$1 tiles=${2:-16384}
nvcc=${NVCC:-nvcc}
mkdir -p build "$root"
"$nvcc" --version | tail -1
"$nvcc" -O2 -arch=sm_75 -o build/mma_sm70_on_sm75 gpu/mma_sm70.cu
sha256sum build/mma_sm70_on_sm75
insts="m8n8k4.f32.f16.f16.f32 m8n8k4.f32.f16.f16.f16 m8n8k4.f16.f16.f16.f16"
bash tools/gpu_batch.sh gen Turing "$root/main" "$tiles" $insts
python tools/probe_ezero.py gen Turing m8n8k4.f32.f16.f16.f16 "$root/ezero/m8n8k4.f32.f16.f16.f16"
for i in $insts; do python tools/probe_f16_round.py gen Turing "$i" "$root/round/$i"; done
for set in main ezero round; do
    bash tools/gpu_batch.sh run "$root/$set" /bin/false "$PWD/build/mma_sm70_on_sm75"
done
bash tools/gpu_batch.sh check "$root/main" 8
python tools/probe_ezero.py analyze "$root/ezero/m8n8k4.f32.f16.f16.f16" | grep e_zero_consistent || true
for i in $insts; do python tools/probe_f16_round.py analyze "$root/round/$i" | head -1; done
