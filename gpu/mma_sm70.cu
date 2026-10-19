// Raw-bit harness for Volta's mma.sync.aligned.m8n8k4 (HMMA.884).
//
// On sm_70 one m8n8k4 instruction is four independent 8x8x4 MMAs, one per quad-pair:
// quad-pair q is lanes 4q..4q+3 and 16+4q..16+4q+3 (PTX ISA, "Matrix Fragments for
// mma.m8n8k4 with .f16"). Each quad-pair here gets its own tile, so a warp checks four
// tiles and every quad-pair's unit is exercised.
//
// Files (little endian, tile-major, no header):
//   A: ntiles x 8 x 4 f16, row major        B: ntiles x 8 x 4 f16, B^T row major (n, k)
//   C: ntiles x 8 x 8, f32 or f16, row major D: ntiles x 8 x 8, f32 or f16, row major
//
//   mma_sm70 <variant> <ntiles> <A> <B> <C> <D>
//   variant: f32.f16.f16.f32 | f32.f16.f16.f16 | f16.f16.f16.f16   (D.A.B.C)

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

enum Variant { F32_F32 = 0, F32_F16 = 1, F16_F16 = 2 };  // D type, C type

__device__ inline uint32_t pack2(const uint16_t *p) { return p[0] | (uint32_t(p[1]) << 16); }

template <int V>
__global__ void mma884(const uint16_t *A, const uint16_t *B, const void *C, void *D, int ntiles) {
    const int lane = threadIdx.x % 32;
    const int qp = (lane % 16) / 4;  // quad-pair
    const int tid = lane % 4;        // thread in group
    const int hi = lane / 16;        // second half of the quad-pair: rows/cols 4..7
    const long tile = (long(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32) * 4 + qp;
    const bool live = tile < ntiles;
    const long t = live ? tile : 0;  // dead lanes still execute mma.sync (it is warp-wide)

    // A (row): a_i = A[row][i], row = tid + 4*hi. B (col): b_i = B[i][col], col = tid + 4*hi.
    const int rc = tid + 4 * hi;
    const uint16_t *a = A + (t * 8 + rc) * 4;
    const uint16_t *b = B + (t * 8 + rc) * 4;  // B^T row rc = column rc of B
    uint32_t a0 = pack2(a), a1 = pack2(a + 2), b0 = pack2(b), b1 = pack2(b + 2);

    if (V == F16_F16) {
        // C/D f16: c_i at row rc, col i; four registers of two halves.
        const uint16_t *c = (const uint16_t *)C + (t * 8 + rc) * 8;
        uint32_t c0 = pack2(c), c1 = pack2(c + 2), c2 = pack2(c + 4), c3 = pack2(c + 6);
        uint32_t d0, d1, d2, d3;
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16 "
                     "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11};"
                     : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                     : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(c0), "r"(c1), "r"(c2), "r"(c3));
        if (live) {
            uint32_t *d = (uint32_t *)((uint16_t *)D + (t * 8 + rc) * 8);
            d[0] = d0, d[1] = d1, d[2] = d2, d[3] = d3;
        }
        return;
    }

    // C/D f32: c_i at row (tid % 2) + (i & 2) + 4*hi, col (i & 4) + (tid & 2) + (i & 1).
    float c[8], d[8];
    if (V == F32_F32) {
        for (int i = 0; i < 8; i++) {
            int row = tid % 2 + (i & 2) + 4 * hi, col = (i & 4) + (tid & 2) + (i & 1);
            c[i] = ((const float *)C)[(t * 8 + row) * 8 + col];
        }
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};"
                     : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
                     : "r"(a0), "r"(a1), "r"(b0), "r"(b1),
                       "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "f"(c[4]), "f"(c[5]), "f"(c[6]), "f"(c[7]));
    } else {
        const uint16_t *ch = (const uint16_t *)C + (t * 8 + rc) * 8;  // C f16, as in F16_F16
        uint32_t c0 = pack2(ch), c1 = pack2(ch + 2), c2 = pack2(ch + 4), c3 = pack2(ch + 6);
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f16 "
                     "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15};"
                     : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
                     : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(c0), "r"(c1), "r"(c2), "r"(c3));
    }
    if (live) {
        for (int i = 0; i < 8; i++) {
            int row = tid % 2 + (i & 2) + 4 * hi, col = (i & 4) + (tid & 2) + (i & 1);
            ((float *)D)[(t * 8 + row) * 8 + col] = d[i];
        }
    }
}

static std::vector<char> slurp(const char *path, size_t n) {
    std::vector<char> v(n);
    FILE *f = fopen(path, "rb");
    if (!f || fread(v.data(), 1, n, f) != n) {
        fprintf(stderr, "cannot read %zu bytes from %s\n", n, path);
        exit(2);
    }
    fclose(f);
    return v;
}

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s <D.A.B.C> <ntiles> <A> <B> <C> <D>\n", argv[0]);
        return 2;
    }
    int v = !strcmp(argv[1], "f32.f16.f16.f32") ? F32_F32
          : !strcmp(argv[1], "f32.f16.f16.f16") ? F32_F16
          : !strcmp(argv[1], "f16.f16.f16.f16") ? F16_F16 : -1;
    if (v < 0) {
        fprintf(stderr, "unknown variant %s\n", argv[1]);
        return 2;
    }
    const long nt = atol(argv[2]);
    const size_t c_bytes = v == F32_F32 ? 4 : 2, d_bytes = v == F16_F16 ? 2 : 4;
    auto a = slurp(argv[3], nt * 32 * 2), b = slurp(argv[4], nt * 32 * 2), c = slurp(argv[5], nt * 64 * c_bytes);
    void *dA, *dB, *dC, *dD;
    CHECK(cudaMalloc(&dA, a.size()));
    CHECK(cudaMalloc(&dB, b.size()));
    CHECK(cudaMalloc(&dC, c.size()));
    CHECK(cudaMalloc(&dD, nt * 64 * d_bytes));
    CHECK(cudaMemcpy(dA, a.data(), a.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, b.data(), b.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dC, c.data(), c.size(), cudaMemcpyHostToDevice));
    const int threads = 256, tiles_per_block = threads / 32 * 4;
    const int blocks = int((nt + tiles_per_block - 1) / tiles_per_block);
    auto A = (const uint16_t *)dA, B = (const uint16_t *)dB;
    if (v == F32_F32) mma884<F32_F32><<<blocks, threads>>>(A, B, dC, dD, int(nt));
    if (v == F32_F16) mma884<F32_F16><<<blocks, threads>>>(A, B, dC, dD, int(nt));
    if (v == F16_F16) mma884<F16_F16><<<blocks, threads>>>(A, B, dC, dD, int(nt));
    CHECK(cudaGetLastError());
    std::vector<char> d(nt * 64 * d_bytes);
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
    printf("{\"device\": \"%s\", \"sm\": %d%d, \"variant\": \"%s\", \"tiles\": %ld}\n", p.name, p.major, p.minor,
           argv[1], nt);
    return 0;
}
