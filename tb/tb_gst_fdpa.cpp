// Verilator testbench for rtl/gst_fdpa.sv: reads vectors from tools/bs_rtl_vectors.py
// (a[0..K-1] b[0..K-1] c sfa[0..NB-1] sfb[0..NB-1] d, hex) and compares every output bit.
// Exit status 1 on any mismatch.
//   tb <vectors.txt> [max mismatches to print]
// Build with -DTB_K=<elements> -DTB_NB=<scale factors> -DTB_LAT=<0 or 8>.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "Vgst_fdpa.h"
#include "verilated.h"

#ifndef TB_K
#define TB_K 64
#endif
#ifndef TB_NB
#define TB_NB 2
#endif
#ifndef TB_LAT
#define TB_LAT 0
#endif

struct Case {
    unsigned a[TB_K], b[TB_K], c, sfa[TB_NB], sfb[TB_NB], d;
};

// K nibbles into a packed [K-1:0][3:0] port (a VlWide of 32-bit words).
template <typename T>
void put_nibbles(T &port, const unsigned *x) {
    for (int i = 0; i < TB_K / 8; i++) {
        uint32_t w = 0;
        for (int j = 0; j < 8; j++) w |= (x[8 * i + j] & 0xFu) << (4 * j);
        port[i] = w;
    }
}

unsigned bytes(const unsigned *x) {
    unsigned v = 0;
    for (int j = 0; j < TB_NB; j++) v |= (x[j] & 0xFFu) << (8 * j);
    return v;
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
        for (int k = 0; k < TB_K; k++) got += fscanf(f, "%x", &x.a[k]);
        for (int k = 0; k < TB_K; k++) got += fscanf(f, "%x", &x.b[k]);
        got += fscanf(f, "%x", &x.c);
        for (int j = 0; j < TB_NB; j++) got += fscanf(f, "%x", &x.sfa[j]);
        for (int j = 0; j < TB_NB; j++) got += fscanf(f, "%x", &x.sfb[j]);
        got += fscanf(f, "%x", &x.d);
        if (got != 2 * TB_K + 2 * TB_NB + 2) break;
        cases.push_back(x);
    }
    fclose(f);
    const long show = argc > 2 ? atol(argv[2]) : 10;
    const long n = long(cases.size());
    auto *dut = new Vgst_fdpa;
    long bad = 0;

    auto check = [&](long i) {
        const Case &x = cases[i];
        if (dut->d == x.d) return;
        if (bad++ < show)
            fprintf(stderr, "case %ld: c=%08x sfa0=%02x sfb0=%02x expected %08x got %08x\n", i, x.c,
                    x.sfa[0], x.sfb[0], x.d, dut->d);
    };

    static const Case zero = {};
    dut->ce = 1;
    dut->clk = 0;
    dut->eval();  // without this the first rising edge is not seen
    for (long j = 0; j < n + TB_LAT; j++) {
        const Case &x = j < n ? cases[j] : zero;
        put_nibbles(dut->a, x.a);
        put_nibbles(dut->b, x.b);
        dut->c = x.c;
        dut->sfa = bytes(x.sfa);
        dut->sfb = bytes(x.sfb);
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
