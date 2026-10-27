# Tensor-core throughput, FP16 vs FP32 accumulate (RTX 5070 Ti, RTX 2060 Super)

Source: `gpu/tc_throughput.cu`, runner `tools/run_tc_throughput.sh`, commit 4efa3c9.
Host: Ryzen 9 9950X3D desktop, Windows 11 + WSL2, driver 621.01, CUDA 13.3.
GPU0 RTX 5070 Ti drives the display; GPU1 RTX 2060 Super is headless.

Files:
- `dev0.txt`, `dev1.txt`, `smi.csv`, `gpu.csv`, `host.txt`: run 2, produced by the runner at 4efa3c9.
- `run1-precommit.txt`: run 1, same kernel code, idle desktop (transcribed).

| GPU | Measure | FP16 acc | FP32 acc | Ratio |
|---|---|---|---|---|
| 5070 Ti | mma.sync m16n8k16, run 1 | 199.65 | 101.53 | 1.97 |
| 5070 Ti | cuBLAS 8192^3, run 1 | 191.89 | 98.00 | 1.96 |
| 5070 Ti | mma.sync m16n8k16, run 2 | 177.58 | 96.31 | 1.84 |
| 5070 Ti | cuBLAS 8192^3, run 2 | 141.22 | 81.15 | 1.74 |
| 2060 Super | mma.sync m16n8k8, run 1 / run 2 | 68.80 / 69.39 | 34.45 / 34.45 | 2.00 / 2.01 |
| 2060 Super | cuBLAS 8192^3, run 1 / run 2 | 60.86 / 60.44 | 33.36 / 33.35 | 1.82 / 1.81 |

Run 2 on the 5070 Ti shared the GPU with an active Windows desktop session (browser, overlays; 13 % GPU
utilisation idle afterwards) at the same 2827-2835 MHz, so its absolute numbers and ratios are
lower; run 1 and the headless 2060 Super are the clean ones. Clocks were not locked.
