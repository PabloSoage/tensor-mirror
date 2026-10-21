// Verilator testbench for rtl/tfdpa.sv: reads vectors from tools/rtl_vectors.py and
// compares every output bit. Exit status 1 on any mismatch.
//   tb <vectors.txt> [max mismatches to print]
// Build with -DTB_L=<terms> -DTB_AW=<A word bits> -DTB_BW=<B word bits> (tools/rtl_params.py)
// and -DTB_LAT=<latency> (0 for PIPE = 0, 8 for PIPE = 1): one case per cycle, and the
// result of case i is checked LAT cycles later. With -DTB_ST (ST = 1) each line also has
// sfa and sfb after c: a[0..L-1] b[0..L-1] c sfa sfb d.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

#include "Vtfdpa.h"
#include "verilated.h"

#ifndef TB_L
#define TB_L 4
#endif
#ifndef TB_AW
#define TB_AW 16
#endif
#ifndef TB_BW
#define TB_BW 16
#endif
#ifndef TB_LAT
#define TB_LAT 0
#endif
#ifndef TB_ST
#define TB_ST 0
#endif

struct Case {
    unsigned a[TB_L], b[TB_L], c, sfa, sfb, d;
};

// Packs x[0..L-1], w bits each, into a packed [L-1:0][w-1:0] port: an integer up to 64
// bits, a VlWide (32-bit words) above that.
template <typename T>
void put(T &port, const unsigned *x, int w) {
    if constexpr (std::is_integral_v<T>) {
        uint64_t v = 0;
        for (int k = 0; k < TB_L; k++) v |= uint64_t(x[k]) << (w * k);
        port = T(v);
    } else {
        for (int i = 0; i < (TB_L * w + 31) / 32; i++) port[i] = 0;
        for (int k = 0; k < TB_L; k++)
            for (int j = 0; j < w; j++)
                if ((x[k] >> j) & 1u) port[(w * k + j) / 32] |= 1u << ((w * k + j) % 32);
    }
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    if (argc < 2) {
        fprintf(stderr, "usage: %s <vectors.txt> [show]\n", argv[0]);
        return 2;
    }
    FILE *f = fopen(argv[1], "r");
    if (!f) {
        perror(argv[1]);
        return 2;
    }
    std::vector<Case> cases;
    for (Case x;;) {
        int got = 0;
        for (int k = 0; k < TB_L; k++) got += fscanf(f, "%x", &x.a[k]);
        for (int k = 0; k < TB_L; k++) got += fscanf(f, "%x", &x.b[k]);
        got += fscanf(f, "%x", &x.c);
        x.sfa = x.sfb = 0;
        if (TB_ST) got += fscanf(f, "%x %x", &x.sfa, &x.sfb);
        got += fscanf(f, "%x", &x.d);
        if (got != 2 * TB_L + 2 + (TB_ST ? 2 : 0)) break;
        cases.push_back(x);
    }
    fclose(f);
    const long show = argc > 2 ? atol(argv[2]) : 10;
    const long n = long(cases.size());
    auto *dut = new Vtfdpa;
    long bad = 0;

    auto check = [&](long i) {
        const Case &x = cases[i];
        if (dut->d == x.d) return;
        if (bad++ < show) {
            fprintf(stderr, "case %ld: a=", i);
            for (int k = 0; k < TB_L; k++) fprintf(stderr, "%x ", x.a[k]);
            fprintf(stderr, "b=");
            for (int k = 0; k < TB_L; k++) fprintf(stderr, "%x ", x.b[k]);
            fprintf(stderr, "c=%08x expected %08x got %08x\n", x.c, x.d, dut->d);
        }
    };

    const unsigned zeros[TB_L] = {};
    dut->ce = 1;
    dut->clk = 0;
    dut->eval();  // without this the first rising edge is not seen
    for (long j = 0; j < n + TB_LAT; j++) {
        const bool in = j < n;
        put(dut->a, in ? cases[j].a : zeros, TB_AW);
        put(dut->b, in ? cases[j].b : zeros, TB_BW);
        dut->c = in ? cases[j].c : 0;
        dut->sfa = in ? cases[j].sfa : 0;
        dut->sfb = in ? cases[j].sfb : 0;
        if (TB_LAT == 0) {
            if (!in) break;
            dut->eval();
            check(j);
            continue;
        }
        dut->clk = 1;  // the edge that takes case j in; d now holds case j - (LAT - 1)
        dut->eval();
        if (j >= TB_LAT - 1 && j - (TB_LAT - 1) < n) check(j - (TB_LAT - 1));
        dut->clk = 0;
        dut->eval();
    }
    printf("{\"cases\": %ld, \"latency\": %d, \"mismatches\": %ld}\n", n, TB_LAT, bad);
    delete dut;
    return bad || n == 0 ? 1 : 0;
}
