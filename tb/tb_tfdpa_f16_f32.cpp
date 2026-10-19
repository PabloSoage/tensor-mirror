// Verilator testbench for rtl/tfdpa_f16_f32.sv: reads vectors from tools/rtl_vectors.py
// and compares every output bit. Exit status 1 on any mismatch.
//   tb <vectors.txt> [max mismatches to print]

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "Vtfdpa_f16_f32.h"
#include "verilated.h"

#ifndef TB_L
#define TB_L 4
#endif

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
    const long show = argc > 2 ? atol(argv[2]) : 10;
    auto *dut = new Vtfdpa_f16_f32;
    long n = 0, bad = 0;
    unsigned a[TB_L], b[TB_L], c, d;
    for (;;) {
        int got = 0;
        for (int k = 0; k < TB_L; k++) got += fscanf(f, "%x", &a[k]);
        for (int k = 0; k < TB_L; k++) got += fscanf(f, "%x", &b[k]);
        got += fscanf(f, "%x %x", &c, &d);
        if (got != 2 * TB_L + 2) break;
        // a and b are packed [L-1:0][15:0]; with L = 4 Verilator holds them as a uint64_t
        static_assert(TB_L == 4, "the packing below assumes L = 4");
        uint64_t pa = 0, pb = 0;
        for (int k = 0; k < TB_L; k++) {
            pa |= uint64_t(a[k]) << (16 * k);
            pb |= uint64_t(b[k]) << (16 * k);
        }
        dut->a = pa;
        dut->b = pb;
        dut->c = c;
        dut->eval();
        if (dut->d != d) {
            if (bad < show) {
                fprintf(stderr, "case %ld: a=", n);
                for (int k = 0; k < TB_L; k++) fprintf(stderr, "%04x ", a[k]);
                fprintf(stderr, "b=");
                for (int k = 0; k < TB_L; k++) fprintf(stderr, "%04x ", b[k]);
                fprintf(stderr, "c=%08x expected %08x got %08x\n", c, d, dut->d);
            }
            bad++;
        }
        n++;
    }
    fclose(f);
    printf("{\"cases\": %ld, \"mismatches\": %ld}\n", n, bad);
    delete dut;
    return bad || n == 0 ? 1 : 0;
}
