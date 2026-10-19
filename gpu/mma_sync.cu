// Raw-bit harness for mma.sync.aligned.m16n8kK.row.col (Turing to RTX Blackwell).
//
// One warp computes one 16 x 8 x K tile. Fragment layouts follow the PTX ISA ("Matrix
// Fragments for mma.m16n8kK"), written per element size; they agree with MMA-Sim's
// mmasim-kernels/mma.h for the shapes both cover. Elements of 1 byte hold FP8, FP6 (bits
// 5:0) or FP4 (bits 5:2) as the instruction reads them.
//
// Files (little endian, tile-major, no header):
//   A: ntiles x 16 x K, row major        B: ntiles x 8 x K (B^T, row major)
//   C: ntiles x 16 x 8, row major        D: ntiles x 16 x 8, row major
//
//   mma_sync --list
//   mma_sync <instruction> <ntiles> <A> <B> <C> <D>    e.g. m16n8k16.f32.bf16.bf16.f32
//
// Build: nvcc -O2 -arch=sm_75 (Turing set) | -arch=sm_80 | -arch=sm_89 | -arch=sm_120a (all)

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x)                                                                       \
    do {                                                                               \
        cudaError_t e_ = (x);                                                          \
        if (e_ != cudaSuccess) {                                                       \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            exit(2);                                                                   \
        }                                                                              \
    } while (0)

// ---- the instructions: K, bytes per element of A, B, C, D, and the asm ---------------------
// Register counts: A holds 16*K/32 elements, B 8*K/32, C and D 4 (f32: 4 regs, f16: 2).
#define MMA_F32(NAME, PTX, K_, EA_, EB_, NA, NB)                                           \
    struct NAME {                                                                          \
        static constexpr int K = K_, EA = EA_, EB = EB_, EC = 4, ED = 4;                   \
        static constexpr const char *name = PTX;                                           \
        __device__ static void run(const uint32_t *a, const uint32_t *b, const uint32_t *c, \
                                   uint32_t *d) {                                          \
            float f0, f1, f2, f3;                                                          \
            MMA_ASM_##NA##_##NB(PTX, "f", f0, f1, f2, f3, __uint_as_float(c[0]),           \
                                __uint_as_float(c[1]), __uint_as_float(c[2]),              \
                                __uint_as_float(c[3]));                                    \
            d[0] = __float_as_uint(f0), d[1] = __float_as_uint(f1);                        \
            d[2] = __float_as_uint(f2), d[3] = __float_as_uint(f3);                        \
        }                                                                                  \
    };
#define MMA_F16(NAME, PTX, K_, EA_, EB_, NA, NB)                                           \
    struct NAME {                                                                          \
        static constexpr int K = K_, EA = EA_, EB = EB_, EC = 2, ED = 2;                   \
        static constexpr const char *name = PTX;                                           \
        __device__ static void run(const uint32_t *a, const uint32_t *b, const uint32_t *c, \
                                   uint32_t *d) {                                          \
            MMA_ASM16_##NA##_##NB(PTX, d[0], d[1], c[0], c[1]);                            \
        }                                                                                  \
    };
#define MMA_ASM_2_1(PTX, T, d0, d1, d2, d3, c0, c1, c2, c3)                                \
    asm volatile("mma.sync.aligned." PTX " {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"  \
                 : "=" T(d0), "=" T(d1), "=" T(d2), "=" T(d3)                              \
                 : "r"(a[0]), "r"(a[1]), "r"(b[0]), T(c0), T(c1), T(c2), T(c3))
#define MMA_ASM_4_2(PTX, T, d0, d1, d2, d3, c0, c1, c2, c3)                                \
    asm volatile("mma.sync.aligned." PTX                                                   \
                 " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"              \
                 : "=" T(d0), "=" T(d1), "=" T(d2), "=" T(d3)                              \
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),        \
                   T(c0), T(c1), T(c2), T(c3))
#define MMA_ASM16_2_1(PTX, d0, d1, c0, c1)                                                  \
    asm volatile("mma.sync.aligned." PTX " {%0,%1}, {%2,%3}, {%4}, {%5,%6};"               \
                 : "=r"(d0), "=r"(d1) : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(c0), "r"(c1))
#define MMA_ASM16_4_2(PTX, d0, d1, c0, c1)                                                  \
    asm volatile("mma.sync.aligned." PTX " {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};"      \
                 : "=r"(d0), "=r"(d1)                                                      \
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),        \
                   "r"(c0), "r"(c1))

#ifndef TM_SM
#error "build with -DTM_SM=75, 80, 89 or 120 (matching -arch)"
#endif
// sm_75
MMA_F32(I_k8_f32_f16, "m16n8k8.row.col.f32.f16.f16.f32", 8, 2, 2, 2, 1)
MMA_F16(I_k8_f16_f16, "m16n8k8.row.col.f16.f16.f16.f16", 8, 2, 2, 2, 1)
#if TM_SM >= 80
MMA_F32(I_k16_f32_f16, "m16n8k16.row.col.f32.f16.f16.f32", 16, 2, 2, 4, 2)
MMA_F16(I_k16_f16_f16, "m16n8k16.row.col.f16.f16.f16.f16", 16, 2, 2, 4, 2)
MMA_F32(I_k16_f32_bf16, "m16n8k16.row.col.f32.bf16.bf16.f32", 16, 2, 2, 4, 2)
MMA_F32(I_k8_f32_bf16, "m16n8k8.row.col.f32.bf16.bf16.f32", 8, 2, 2, 2, 1)
MMA_F32(I_k8_f32_tf32, "m16n8k8.row.col.f32.tf32.tf32.f32", 8, 4, 4, 4, 2)
MMA_F32(I_k4_f32_tf32, "m16n8k4.row.col.f32.tf32.tf32.f32", 4, 4, 4, 2, 1)
#endif
#if TM_SM >= 89  // FP8, f32 and f16 accumulators, k16 and k32
#define FP8_SET(A, B)                                                                        \
    MMA_F32(I_k32_f32_##A##_##B, "m16n8k32.row.col.f32." #A "." #B ".f32", 32, 1, 1, 4, 2)  \
    MMA_F32(I_k16_f32_##A##_##B, "m16n8k16.row.col.f32." #A "." #B ".f32", 16, 1, 1, 2, 1)  \
    MMA_F16(I_k32_f16_##A##_##B, "m16n8k32.row.col.f16." #A "." #B ".f16", 32, 1, 1, 4, 2)  \
    MMA_F16(I_k16_f16_##A##_##B, "m16n8k16.row.col.f16." #A "." #B ".f16", 16, 1, 1, 2, 1)
FP8_SET(e4m3, e4m3)
FP8_SET(e4m3, e5m2)
FP8_SET(e5m2, e4m3)
FP8_SET(e5m2, e5m2)
#endif
#if TM_SM >= 120  // sm_120a: kind::f8f6f4, every pair of e4m3, e5m2, e3m2, e2m3, e2m1
#define F8F6F4(A, B)                                                                          \
    MMA_F32(I_f8f6f4_f32_##A##_##B, "m16n8k32.row.col.kind::f8f6f4.f32." #A "." #B ".f32", 32, \
            1, 1, 4, 2)                                                                       \
    MMA_F16(I_f8f6f4_f16_##A##_##B, "m16n8k32.row.col.kind::f8f6f4.f16." #A "." #B ".f16", 32, \
            1, 1, 4, 2)
#define F8F6F4_ROW(A) F8F6F4(A, e4m3) F8F6F4(A, e5m2) F8F6F4(A, e3m2) F8F6F4(A, e2m3) F8F6F4(A, e2m1)
F8F6F4_ROW(e4m3)
F8F6F4_ROW(e5m2)
F8F6F4_ROW(e3m2)
F8F6F4_ROW(e2m3)
F8F6F4_ROW(e2m1)
#endif

// ---- fragment positions -----------------------------------------------------------------
// g = lane / 4 (group), t = lane % 4 (thread in group); i = element index in the fragment.
__device__ inline void a_pos(int es, int i, int g, int t, int &row, int &col) {
    if (es == 4) row = g + 8 * (i & 1), col = t + 4 * (i >> 1);
    else if (es == 2) row = g + 8 * ((i >> 1) & 1), col = t * 2 + (i & 1) + 8 * (i >> 2);
    else row = g + 8 * ((i >> 2) & 1), col = t * 4 + (i & 3) + 16 * (i >> 3);
}
__device__ inline int b_pos(int es, int i, int t) {  // k; the column (n) is g
    if (es == 4) return t + 4 * i;
    if (es == 2) return t * 2 + (i & 1) + 8 * (i >> 1);
    return t * 4 + (i & 3) + 16 * (i >> 2);
}
__device__ inline uint32_t load(const uint8_t *p, int es) {
    uint32_t v = 0;
    for (int j = 0; j < es; j++) v |= uint32_t(p[j]) << (8 * j);
    return v;
}

template <class I>
__global__ void tile_kernel(const uint8_t *A, const uint8_t *B, const uint8_t *C, uint8_t *D, int ntiles) {
    constexpr int K = I::K, NA = 16 * K / 32, NB = 8 * K / 32;
    const int lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    const long tile = long(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    const long tt = tile < ntiles ? tile : 0;  // dead warps still run mma.sync
    uint32_t a[4] = {0, 0, 0, 0}, b[2] = {0, 0}, c[4] = {0, 0, 0, 0}, d[4];
    for (int i = 0; i < NA; i++) {
        int row, col;
        a_pos(I::EA, i, g, t, row, col);
        a[i * I::EA / 4] |= load(A + ((tt * 16 + row) * K + col) * I::EA, I::EA) << (8 * (i * I::EA % 4));
    }
    for (int i = 0; i < NB; i++) {
        int k = b_pos(I::EB, i, t);
        b[i * I::EB / 4] |= load(B + ((tt * 8 + g) * K + k) * I::EB, I::EB) << (8 * (i * I::EB % 4));
    }
    for (int i = 0; i < 4; i++) {  // C and D: row g + 8 * (i / 2), column 2t + i % 2
        uint32_t v = load(C + ((tt * 16 + g + 8 * (i / 2)) * 8 + 2 * t + i % 2) * I::EC, I::EC);
        if (I::EC == 4) c[i] = v;
        else c[i / 2] |= v << (16 * (i % 2));
    }
    I::run(a, b, c, d);
    if (tile >= ntiles) return;
    for (int i = 0; i < 4; i++) {
        uint32_t v = I::ED == 4 ? d[i] : (d[i / 2] >> (16 * (i % 2))) & 0xffff;
        uint8_t *p = D + ((tile * 16 + g + 8 * (i / 2)) * 8 + 2 * t + i % 2) * I::ED;
        for (int j = 0; j < I::ED; j++) p[j] = (v >> (8 * j)) & 0xff;
    }
}

// ---- host ----------------------------------------------------------------------------------
struct Entry {
    const char *name;  // shape.D.A.B.C as tensormirror and MMA-Sim name it
    int K, EA, EB, EC, ED;
    void (*launch)(const uint8_t *, const uint8_t *, const uint8_t *, uint8_t *, int);
};

template <class I>
static void launch(const uint8_t *A, const uint8_t *B, const uint8_t *C, uint8_t *D, int nt) {
    const int warps = 8;
    tile_kernel<I><<<(nt + warps - 1) / warps, 32 * warps>>>(A, B, C, D, nt);
}
#define E(I, NAME) {NAME, I::K, I::EA, I::EB, I::EC, I::ED, launch<I>},
#define FP8_ENTRIES(A, B)                                                  \
    E(I_k32_f32_##A##_##B, "m16n8k32.f32." #A "." #B ".f32")               \
    E(I_k16_f32_##A##_##B, "m16n8k16.f32." #A "." #B ".f32")               \
    E(I_k32_f16_##A##_##B, "m16n8k32.f16." #A "." #B ".f16")               \
    E(I_k16_f16_##A##_##B, "m16n8k16.f16." #A "." #B ".f16")
#define F8F6F4_ENTRIES(A, B)                                               \
    E(I_f8f6f4_f32_##A##_##B, "m16n8k32.f8f6f4.f32." #A "." #B ".f32")     \
    E(I_f8f6f4_f16_##A##_##B, "m16n8k32.f8f6f4.f16." #A "." #B ".f16")
#define F8F6F4_ENTRY_ROW(A)                                                                   \
    F8F6F4_ENTRIES(A, e4m3) F8F6F4_ENTRIES(A, e5m2) F8F6F4_ENTRIES(A, e3m2) F8F6F4_ENTRIES(A, e2m3) \
    F8F6F4_ENTRIES(A, e2m1)

static const Entry table[] = {
#if TM_SM >= 75
    E(I_k8_f32_f16, "m16n8k8.f32.f16.f16.f32")
    E(I_k8_f16_f16, "m16n8k8.f16.f16.f16.f16")
#endif
#if TM_SM >= 80
    E(I_k16_f32_f16, "m16n8k16.f32.f16.f16.f32")
    E(I_k16_f16_f16, "m16n8k16.f16.f16.f16.f16")
    E(I_k16_f32_bf16, "m16n8k16.f32.bf16.bf16.f32")
    E(I_k8_f32_bf16, "m16n8k8.f32.bf16.bf16.f32")
    E(I_k8_f32_tf32, "m16n8k8.f32.tf32.tf32.f32")
    E(I_k4_f32_tf32, "m16n8k4.f32.tf32.tf32.f32")
#endif
#if TM_SM >= 89
    FP8_ENTRIES(e4m3, e4m3) FP8_ENTRIES(e4m3, e5m2) FP8_ENTRIES(e5m2, e4m3) FP8_ENTRIES(e5m2, e5m2)
#endif
#if TM_SM >= 120
    F8F6F4_ENTRY_ROW(e4m3) F8F6F4_ENTRY_ROW(e5m2) F8F6F4_ENTRY_ROW(e3m2) F8F6F4_ENTRY_ROW(e2m3)
    F8F6F4_ENTRY_ROW(e2m1)
#endif
};

static std::vector<uint8_t> slurp(const char *path, size_t n) {
    std::vector<uint8_t> v(n);
    FILE *f = fopen(path, "rb");
    if (!f || fread(v.data(), 1, n, f) != n) {
        fprintf(stderr, "cannot read %zu bytes from %s\n", n, path);
        exit(2);
    }
    fclose(f);
    return v;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--list")) {
        for (const Entry &e : table) printf("%s\n", e.name);
        return 0;
    }
    if (argc != 7) {
        fprintf(stderr, "usage: %s --list | <instruction> <ntiles> <A> <B> <C> <D>\n", argv[0]);
        return 2;
    }
    const Entry *e = nullptr;
    for (const Entry &x : table)
        if (!strcmp(x.name, argv[1])) e = &x;
    if (!e) {
        fprintf(stderr, "instruction %s not in this build (see --list)\n", argv[1]);
        return 2;
    }
    const long nt = atol(argv[2]);
    auto a = slurp(argv[3], nt * 16 * e->K * e->EA), b = slurp(argv[4], nt * 8 * e->K * e->EB),
         c = slurp(argv[5], nt * 128 * e->EC);
    uint8_t *dA, *dB, *dC, *dD;
    CHECK(cudaMalloc(&dA, a.size()));
    CHECK(cudaMalloc(&dB, b.size()));
    CHECK(cudaMalloc(&dC, c.size()));
    CHECK(cudaMalloc(&dD, nt * 128 * e->ED));
    CHECK(cudaMemcpy(dA, a.data(), a.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, b.data(), b.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dC, c.data(), c.size(), cudaMemcpyHostToDevice));
    e->launch(dA, dB, dC, dD, int(nt));
    CHECK(cudaGetLastError());
    std::vector<uint8_t> d(nt * 128 * e->ED);
    CHECK(cudaMemcpy(d.data(), dD, d.size(), cudaMemcpyDeviceToHost));
    FILE *f = fopen(argv[6], "wb");
    if (!f || fwrite(d.data(), 1, d.size(), f) != d.size()) {
        fprintf(stderr, "cannot write %s\n", argv[6]);
        return 2;
    }
    fclose(f);
    cudaDeviceProp p;
    int dev;
    CHECK(cudaGetDevice(&dev));
    CHECK(cudaGetDeviceProperties(&p, dev));
    printf("{\"device\": \"%s\", \"sm\": %d%d, \"instruction\": \"%s\", \"tiles\": %ld}\n", p.name, p.major, p.minor,
           argv[1], nt);
    return 0;
}
