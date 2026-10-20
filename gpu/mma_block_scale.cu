// Raw-bit harness for mma.sync.aligned.m16n8kK.row.col.kind::{mxf8f6f4,mxf4,mxf4nvf4}.block_scale
// (sm_120a).
//
// One warp computes one 16 x 8 x K tile. The A, B, C and D fragments follow the PTX ISA
// ("Matrix Fragments for mma.m16n8k32" with 8-bit containers, "... for mma.m16n8k64" with
// .e2m1, eight elements per register, low to high). The scale operands are taken as they
// are: each lane's scale-a-data and scale-b-data words and the tile's four selectors come
// from the files, so where the scale factors sit is decided (and probed) by the caller.
//
// Files (little endian, tile-major, no header):
//   A: ntiles x 16 x K bytes, row major       B: ntiles x 8 x K bytes (B^T, row major)
//      one element per byte: FP8, FP6 (bits 5:0) or FP4 (bits 5:2) for mxf8f6f4; for the
//      k64 kinds E2M1 in bits 3:0 (the harness packs two per byte; bits 7:4 are dropped)
//   C, D: ntiles x 16 x 8 f32, row major
//   SA, SB: ntiles x 32 words, the lane's scale-a-data / scale-b-data
//   SEL: ntiles x 4 u16: byte-id-a, thread-id-a, byte-id-b, thread-id-b
//
//   mma_block_scale --list
//   mma_block_scale <instruction> <ntiles> <A> <B> <C> <SA> <SB> <SEL> <D>
//
// Build: nvcc -O2 -gencode arch=compute_120a,code=sm_120a

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

#define BS_INST(NAME, PTX, K_)                                                                  \
    struct NAME {                                                                               \
        static constexpr int K = K_;                                                            \
        __device__ static void run(const uint32_t *a, const uint32_t *b, const float *c, float *d, \
                                   uint32_t sa, uint32_t sb, const uint16_t *sel) {             \
            asm volatile("mma.sync.aligned." PTX                                                \
                         " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13},"           \
                         " %14, {%15,%16}, %17, {%18,%19};"                                     \
                         : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])                       \
                         : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),    \
                           "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(sa), "h"(sel[0]),    \
                           "h"(sel[1]), "r"(sb), "h"(sel[2]), "h"(sel[3]));                     \
        }                                                                                       \
    };

#define MXF8F6F4(A, B)                                                                       \
    BS_INST(I_mx_##A##_##B,                                                                  \
            "m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32." #A "." #B ".f32.ue8m0", 32)
#define MXF8F6F4_ROW(A) MXF8F6F4(A, e4m3) MXF8F6F4(A, e5m2) MXF8F6F4(A, e3m2) MXF8F6F4(A, e2m3) MXF8F6F4(A, e2m1)
MXF8F6F4_ROW(e4m3)
MXF8F6F4_ROW(e5m2)
MXF8F6F4_ROW(e3m2)
MXF8F6F4_ROW(e2m3)
MXF8F6F4_ROW(e2m1)
BS_INST(I_mxf4_2x, "m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0", 64)
BS_INST(I_nvf4_2x_ue8m0, "m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0", 64)
#ifdef TM_TRY_4X_UE8M0  // in the PTX ISA table, but ptxas of CUDA 12.9 rejects it
BS_INST(I_nvf4_4x_ue8m0, "m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue8m0", 64)
#endif
BS_INST(I_nvf4_4x_ue4m3, "m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3", 64)
#ifdef TM_TRY_2X_UE4M3  // not in the PTX ISA's table of valid combinations (H-010, prediction 6)
BS_INST(I_nvf4_2x_ue4m3, "m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue4m3", 64)
#endif

// ---- fragment positions -----------------------------------------------------------------
// g = lane / 4, t = lane % 4. K = 32: i-th byte of the A (16) and B (8) fragments; K = 64:
// i-th nibble of the A (32) and B (16) fragments.
__device__ inline void a_pos(int K, int i, int g, int t, int &row, int &col) {
    if (K == 32) row = g + 8 * ((i >> 2) & 1), col = t * 4 + (i & 3) + 16 * (i >> 3);
    else row = g + 8 * ((i >> 3) & 1), col = t * 8 + (i & 7) + 32 * (i >> 4);
}
__device__ inline int b_pos(int K, int i, int t) {
    return K == 32 ? t * 4 + (i & 3) + 16 * (i >> 2) : t * 8 + (i & 7) + 32 * (i >> 3);
}

template <class I>
__global__ void tile_kernel(const uint8_t *A, const uint8_t *B, const float *C, const uint32_t *SA,
                            const uint32_t *SB, const uint16_t *SEL, float *D, int ntiles) {
    constexpr int K = I::K, BITS = K == 32 ? 8 : 4, NA = 16 * K / 32, NB = 8 * K / 32;
    const int lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    const long tile = long(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    const long tt = tile < ntiles ? tile : 0;  // dead warps still run mma.sync
    uint32_t a[4] = {0, 0, 0, 0}, b[2] = {0, 0};
    float c[4], d[4];
    for (int i = 0; i < NA; i++) {
        int row, col;
        a_pos(K, i, g, t, row, col);
        uint32_t v = A[(tt * 16 + row) * K + col] & ((1u << BITS) - 1);
        a[i * BITS / 32] |= v << (i * BITS % 32);
    }
    for (int i = 0; i < NB; i++) {
        uint32_t v = B[(tt * 8 + g) * K + b_pos(K, i, t)] & ((1u << BITS) - 1);
        b[i * BITS / 32] |= v << (i * BITS % 32);
    }
    for (int i = 0; i < 4; i++) c[i] = C[(tt * 16 + g + 8 * (i / 2)) * 8 + 2 * t + i % 2];
    I::run(a, b, c, d, SA[tt * 32 + lane], SB[tt * 32 + lane], SEL + tt * 4);
    if (tile >= ntiles) return;
    for (int i = 0; i < 4; i++) D[(tile * 16 + g + 8 * (i / 2)) * 8 + 2 * t + i % 2] = d[i];
}

// ---- host ----------------------------------------------------------------------------------
struct Entry {
    const char *name;  // shape.kind.scale_vec.D.A.B.C.stype
    int K;
    void (*launch)(const uint8_t *, const uint8_t *, const float *, const uint32_t *, const uint32_t *,
                   const uint16_t *, float *, int);
};

template <class I>
static void launch(const uint8_t *A, const uint8_t *B, const float *C, const uint32_t *SA, const uint32_t *SB,
                   const uint16_t *SEL, float *D, int nt) {
    const int warps = 8;
    tile_kernel<I><<<(nt + warps - 1) / warps, 32 * warps>>>(A, B, C, SA, SB, SEL, D, nt);
}
#define E(I, NAME) {NAME, I::K, launch<I>},
#define MX_ENTRY(A, B) E(I_mx_##A##_##B, "m16n8k32.mxf8f6f4.1x.f32." #A "." #B ".f32.ue8m0")
#define MX_ENTRY_ROW(A) MX_ENTRY(A, e4m3) MX_ENTRY(A, e5m2) MX_ENTRY(A, e3m2) MX_ENTRY(A, e2m3) MX_ENTRY(A, e2m1)

static const Entry table[] = {
    MX_ENTRY_ROW(e4m3) MX_ENTRY_ROW(e5m2) MX_ENTRY_ROW(e3m2) MX_ENTRY_ROW(e2m3) MX_ENTRY_ROW(e2m1)
    E(I_mxf4_2x, "m16n8k64.mxf4.2x.f32.e2m1.e2m1.f32.ue8m0")
    E(I_nvf4_2x_ue8m0, "m16n8k64.mxf4nvf4.2x.f32.e2m1.e2m1.f32.ue8m0")
#ifdef TM_TRY_4X_UE8M0
    E(I_nvf4_4x_ue8m0, "m16n8k64.mxf4nvf4.4x.f32.e2m1.e2m1.f32.ue8m0")
#endif
    E(I_nvf4_4x_ue4m3, "m16n8k64.mxf4nvf4.4x.f32.e2m1.e2m1.f32.ue4m3")
#ifdef TM_TRY_2X_UE4M3
    E(I_nvf4_2x_ue4m3, "m16n8k64.mxf4nvf4.2x.f32.e2m1.e2m1.f32.ue4m3")
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

template <class T>
static T *to_device(const std::vector<uint8_t> &v) {
    void *p;
    CHECK(cudaMalloc(&p, v.size()));
    CHECK(cudaMemcpy(p, v.data(), v.size(), cudaMemcpyHostToDevice));
    return static_cast<T *>(p);
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--list")) {
        for (const Entry &e : table) printf("%s\n", e.name);
        return 0;
    }
    if (argc != 10) {
        fprintf(stderr, "usage: %s --list | <instruction> <ntiles> <A> <B> <C> <SA> <SB> <SEL> <D>\n", argv[0]);
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
    auto *dA = to_device<uint8_t>(slurp(argv[3], nt * 16 * e->K));
    auto *dB = to_device<uint8_t>(slurp(argv[4], nt * 8 * e->K));
    auto *dC = to_device<float>(slurp(argv[5], nt * 128 * 4));
    auto *dSA = to_device<uint32_t>(slurp(argv[6], nt * 32 * 4));
    auto *dSB = to_device<uint32_t>(slurp(argv[7], nt * 32 * 4));
    auto *dSEL = to_device<uint16_t>(slurp(argv[8], nt * 4 * 2));
    float *dD;
    CHECK(cudaMalloc(&dD, nt * 128 * 4));
    e->launch(dA, dB, dC, dSA, dSB, dSEL, dD, int(nt));
    CHECK(cudaGetLastError());
    std::vector<uint8_t> d(nt * 128 * 4);
    CHECK(cudaMemcpy(d.data(), dD, d.size(), cudaMemcpyDeviceToHost));
    FILE *f = fopen(argv[9], "wb");
    if (!f || fwrite(d.data(), 1, d.size(), f) != d.size()) {
        fprintf(stderr, "cannot write %s\n", argv[9]);
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
