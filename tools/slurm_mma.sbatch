#!/bin/bash
# Build the mma.sync harnesses for one GPU architecture and run them on a case archive made
# by gpu_batch.sh gen (packed as <name>.tar with the case root inside), on one GPU of a
# SLURM cluster. The outputs are packed into <name>-out.tar for gpu_batch.sh check.
#
#   sbatch --gres=gpu:1 [cluster options] tools/slurm_mma.sbatch <name> <case root> <sm>
#   e.g. tools/slurm_mma.sbatch a100 gpu/a100 80      (from a directory holding a100.tar,
#                                                       gpu/ and tools/ of tensor-mirror)
# CUDA comes from the environment: load the cluster's CUDA module before sbatch, or set
# MODULES to a list of modules to load here.
#SBATCH -J tensor-mirror-mma
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 00:30:00
set -euo pipefail
name=$1 root=$2 sm=$3
cd "$SLURM_SUBMIT_DIR"
if [ -n "${MODULES:-}" ]; then module --force purge; module load $MODULES; fi
nvcc --version | tail -2
nvidia-smi --query-gpu=name,driver_version,vbios_version,pci.bus_id --format=csv
mkdir -p build
nvcc -O2 -arch="sm_$sm" -DTM_SM="$sm" -o "build/mma_sync_sm$sm" gpu/mma_sync.cu
nvcc -O2 -arch="sm_$sm" -o "build/mma_sm70_on_sm$sm" gpu/mma_sm70.cu
rm -rf "$root"
tar -xf "$name.tar"
bash tools/gpu_batch.sh run "$root" "build/mma_sync_sm$sm" "build/mma_sm70_on_sm$sm"
echo "host $(hostname -s) job $SLURM_JOB_ID" > "$root/host.txt"
tar -cf "$name-out.tar" "$root"/gpu.csv "$root"/host.txt "$root"/*/d.bin "$root"/*/run.json
echo done
