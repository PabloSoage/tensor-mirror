// Tensor-core throughput on one GPU: FP16 accumulate vs FP32 accumulate (plus BF16 and TF32),
// with a register-only mma.sync loop (peak) and cuBLAS GemmEx 8192^3. Usage: tc_throughput [device]
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)
#define CB(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { printf("cuBLAS %d @%d\n", (int)s, __LINE__); exit(1); } } while (0)

constexpr int CH = 8;  // independent accumulator chains per warp

// MODE: 0 f16acc k16, 1 f32acc k16, 2 bf16 f32acc k16, 3 f16acc k8, 4 f32acc k8, 5 tf32 k8
template <int MODE>
__global__ void mma_loop(int iters, unsigned *out) {
  unsigned a[4], b[2];
  for (int i = 0; i < 4; i++) a[i] = 0x10011001u + threadIdx.x * 7 + i;
  for (int i = 0; i < 2; i++) b[i] = 0x10031003u + threadIdx.x * 3 + i;
  unsigned c[CH][4];
  for (int j = 0; j < CH; j++) for (int i = 0; i < 4; i++) c[j][i] = 0;
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int j = 0; j < CH; j++) {
      if constexpr (MODE == 0) {
#if __CUDA_ARCH__ >= 800
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};"
                     : "+r"(c[j][0]), "+r"(c[j][1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
      } else if constexpr (MODE == 1) {
#if __CUDA_ARCH__ >= 800
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                     : "+r"(c[j][0]), "+r"(c[j][1]), "+r"(c[j][2]), "+r"(c[j][3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
      } else if constexpr (MODE == 2) {
#if __CUDA_ARCH__ >= 800
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                     : "+r"(c[j][0]), "+r"(c[j][1]), "+r"(c[j][2]), "+r"(c[j][3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
      } else if constexpr (MODE == 3) {
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1},{%2,%3},{%4},{%0,%1};"
                     : "+r"(c[j][0]), "+r"(c[j][1]) : "r"(a[0]), "r"(a[1]), "r"(b[0]));
      } else if constexpr (MODE == 4) {
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5},{%6},{%0,%1,%2,%3};"
                     : "+r"(c[j][0]), "+r"(c[j][1]), "+r"(c[j][2]), "+r"(c[j][3]) : "r"(a[0]), "r"(a[1]), "r"(b[0]));
      } else if constexpr (MODE == 5) {
#if __CUDA_ARCH__ >= 800
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                     : "+r"(c[j][0]), "+r"(c[j][1]), "+r"(c[j][2]), "+r"(c[j][3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
      }
    }
  }
  unsigned s = 0;
  for (int j = 0; j < CH; j++) for (int i = 0; i < 4; i++) s ^= c[j][i];
  if (s == 0x12345678u) out[0] = s;  // keep results live
}

static const char *names[] = {"FP16 in, FP16 acc (m16n8k16)", "FP16 in, FP32 acc (m16n8k16)", "BF16 in, FP32 acc (m16n8k16)",
                              "FP16 in, FP16 acc (m16n8k8) ", "FP16 in, FP32 acc (m16n8k8) ", "TF32 in, FP32 acc (m16n8k8) "};
static const double flops_per_mma[] = {4096, 4096, 4096, 2048, 2048, 2048};

template <int MODE>
double run_mma(int sms, unsigned *out) {
  int blocks = sms * 8, threads = 256, iters = 20000;
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int w = 0; w < 30; w++) mma_loop<MODE><<<blocks, threads>>>(iters, out);  // warm-up / clocks up
  CK(cudaDeviceSynchronize());
  std::vector<double> t;
  for (int r = 0; r < 9; r++) {
    cudaEventRecord(e0);
    mma_loop<MODE><<<blocks, threads>>>(iters, out);
    cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
    float ms; cudaEventElapsedTime(&ms, e0, e1); t.push_back(ms);
  }
  std::sort(t.begin(), t.end());
  double flops = (double)blocks * (threads / 32) * iters * CH * flops_per_mma[MODE];
  return flops / (t[t.size() / 2] * 1e-3) / 1e12;
}

double run_gemm(cublasHandle_t h, int n, cudaDataType ab, cudaDataType cty, cublasComputeType_t comp) {
  size_t esz_ab = (ab == CUDA_R_32F) ? 4 : 2, esz_c = (cty == CUDA_R_32F) ? 4 : 2;
  void *A, *B, *C;
  CK(cudaMalloc(&A, (size_t)n * n * esz_ab)); CK(cudaMalloc(&B, (size_t)n * n * esz_ab)); CK(cudaMalloc(&C, (size_t)n * n * esz_c));
  CK(cudaMemset(A, 0x11, (size_t)n * n * esz_ab)); CK(cudaMemset(B, 0x11, (size_t)n * n * esz_ab)); CK(cudaMemset(C, 0, (size_t)n * n * esz_c));
  __half ah = __float2half(1.f), bh = __float2half(0.f);
  float af = 1.f, bf = 0.f;
  const void *alpha = (comp == CUBLAS_COMPUTE_16F) ? (const void *)&ah : (const void *)&af;
  const void *beta = (comp == CUBLAS_COMPUTE_16F) ? (const void *)&bh : (const void *)&bf;
  auto go = [&] { CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha, A, ab, n, B, ab, n, beta, C, cty, n, comp, CUBLAS_GEMM_DEFAULT)); };
  for (int w = 0; w < 5; w++) go();
  CK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  std::vector<double> t;
  for (int r = 0; r < 9; r++) {
    cudaEventRecord(e0); go(); cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
    float ms; cudaEventElapsedTime(&ms, e0, e1); t.push_back(ms);
  }
  std::sort(t.begin(), t.end());
  cudaFree(A); cudaFree(B); cudaFree(C);
  return 2.0 * n * n * (double)n / (t[t.size() / 2] * 1e-3) / 1e12;
}

int main(int argc, char **argv) {
  int dev = argc > 1 ? atoi(argv[1]) : 0;
  CK(cudaSetDevice(dev));
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));
  int sm = p.major * 10 + p.minor;
  printf("GPU %d: %s, sm_%d, %d SMs\n", dev, p.name, sm, p.multiProcessorCount);
  unsigned *out; CK(cudaMalloc(&out, 4));
  printf("-- raw mma.sync, median of 9 (TFLOPS)\n");
  if (sm >= 80) {
    printf("%s %8.2f\n", names[0], run_mma<0>(p.multiProcessorCount, out));
    printf("%s %8.2f\n", names[1], run_mma<1>(p.multiProcessorCount, out));
    printf("%s %8.2f\n", names[2], run_mma<2>(p.multiProcessorCount, out));
    printf("%s %8.2f\n", names[5], run_mma<5>(p.multiProcessorCount, out));
  }
  printf("%s %8.2f\n", names[3], run_mma<3>(p.multiProcessorCount, out));
  printf("%s %8.2f\n", names[4], run_mma<4>(p.multiProcessorCount, out));

  int n = 8192;
  cublasHandle_t h; CB(cublasCreate(&h));
  printf("-- cuBLAS GemmEx %d^3, median of 9 (TFLOPS)\n", n);
  printf("FP16 in, FP16 out, COMPUTE_16F       %8.2f\n", run_gemm(h, n, CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_16F));
  printf("FP16 in, FP16 out, COMPUTE_32F       %8.2f\n", run_gemm(h, n, CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F));
  if (sm >= 80) {
    printf("BF16 in, BF16 out, COMPUTE_32F       %8.2f\n", run_gemm(h, n, CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F));
    printf("FP32 in, FP32 out, COMPUTE_32F_TF32  %8.2f\n", run_gemm(h, n, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32));
  }
  printf("FP32 in, FP32 out, COMPUTE_32F (SIMT)%8.2f\n", run_gemm(h, n, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_PEDANTIC));
  cublasDestroy(h);
  return 0;
}
