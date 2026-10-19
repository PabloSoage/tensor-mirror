// Verilator testbench for rtl/tfdpa_f16.sv: reads vectors from tools/rtl_vectors.py
// and compares every output bit. Exit status 1 on any mismatch.
//   tb <vectors.txt> [max mismatches to print]
// Build with -DTB_LAT=<latency> (0 for PIPE = 0, 7 for PIPE = 1): one case per cycle, and
// the result of case i is checked LAT cycles later.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "Vtfdpa_f16.h"
#include "verilated.h"

#ifndef TB_L
#define TB_L 4
#endif
#ifndef TB_LAT
#define TB_LAT 0
#endif

struct Case {
    unsigned a[TB_L], b[TB_L], c, d;
};

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
        got += fscanf(f, "%x %x", &x.c, &x.d);
        if (got != 2 * TB_L + 2) break;
        cases.push_back(x);
    }
    fclose(f);
    const long show = argc > 2 ? atol(argv[2]) : 10;
    const long n = long(cases.size());
    auto *dut = new Vtfdpa_f16;
    long bad = 0;

    auto check = [&](long i) {
        const Case &x = cases[i];
        if (dut->d == x.d) return;
        if (bad++ < show) {
            fprintf(stderr, "case %ld: a=", i);
            for (int k = 0; k < TB_L; k++) fprintf(stderr, "%04x ", x.a[k]);
            fprintf(stderr, "b=");
            for (int k = 0; k < TB_L; k++) fprintf(stderr, "%04x ", x.b[k]);
            fprintf(stderr, "c=%08x expected %08x got %08x\n", x.c, x.d, dut->d);
        }
    };

    // a and b are packed [L-1:0][15:0]; with L = 4 Verilator holds them as a uint64_t
    static_assert(TB_L == 4, "the packing below assumes L = 4");
    dut->clk = 0;
    dut->eval();  // without this the first rising edge is not seen
    for (long j = 0; j < n + TB_LAT; j++) {
        uint64_t pa = 0, pb = 0;
        unsigned c = 0;
        if (j < n) {
            for (int k = 0; k < TB_L; k++) {
                pa |= uint64_t(cases[j].a[k]) << (16 * k);
                pb |= uint64_t(cases[j].b[k]) << (16 * k);
            }
            c = cases[j].c;
        }
        dut->a = pa;
        dut->b = pb;
        dut->c = c;
        if (TB_LAT == 0) {
            if (j >= n) break;
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
