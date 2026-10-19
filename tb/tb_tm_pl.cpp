// Verilator testbench for rtl/tm_pl.v: the cases of tools/pl_vectors.py go in on s_axis
// with random gaps, m_axis is drained with random back-pressure, and every result must
// come out once, in order, with the three units' words and tlast on the last case. Then
// the AXI4-Lite registers are read (ID, LATENCY, counters) and the counters cleared.
//   tb <vectors.txt> [seed] [p(valid) percent] [p(ready) percent]
// Exit status 1 on any mismatch.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "Vtm_pl.h"
#include "verilated.h"

struct Case {
    unsigned a[4], b[4], c, d[3];
};

static Vtm_pl *dut;
static void tick() {
    dut->clk = 1;
    dut->eval();
    dut->clk = 0;
    dut->eval();
}

static uint32_t lite_read(uint32_t addr) {
    dut->s_axi_araddr = addr;
    dut->s_axi_arvalid = 1;
    dut->s_axi_rready = 1;
    for (int i = 0; i < 20; i++) {
        tick();
        if (dut->s_axi_arready) dut->s_axi_arvalid = 0;
        if (dut->s_axi_rvalid) {
            uint32_t v = dut->s_axi_rdata;
            tick();
            dut->s_axi_rready = 0;
            return v;
        }
    }
    fprintf(stderr, "AXI-Lite read of 0x%03x timed out\n", addr);
    exit(1);
}

static void lite_write(uint32_t addr, uint32_t v) {
    dut->s_axi_awaddr = addr;
    dut->s_axi_wdata = v;
    dut->s_axi_wstrb = 0xf;
    dut->s_axi_awvalid = 1;
    dut->s_axi_wvalid = 1;
    dut->s_axi_bready = 1;
    for (int i = 0; i < 20; i++) {
        tick();
        if (dut->s_axi_awready) dut->s_axi_awvalid = 0, dut->s_axi_wvalid = 0;
        if (dut->s_axi_bvalid) {
            tick();
            dut->s_axi_bready = 0;
            return;
        }
    }
    fprintf(stderr, "AXI-Lite write of 0x%03x timed out\n", addr);
    exit(1);
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    if (argc < 2) {
        fprintf(stderr, "usage: %s <vectors.txt> [seed] [p_valid%%] [p_ready%%]\n", argv[0]);
        return 2;
    }
    std::vector<Case> cases;
    FILE *f = fopen(argv[1], "r");
    if (!f) return perror(argv[1]), 2;
    for (Case x;;) {
        int got = 0;
        for (int k = 0; k < 4; k++) got += fscanf(f, "%x", &x.a[k]);
        for (int k = 0; k < 4; k++) got += fscanf(f, "%x", &x.b[k]);
        got += fscanf(f, "%x %x %x %x", &x.c, &x.d[0], &x.d[1], &x.d[2]);
        if (got != 12) break;
        cases.push_back(x);
    }
    fclose(f);
    std::mt19937 rng(argc > 2 ? atoi(argv[2]) : 1);
    const int pv = argc > 3 ? atoi(argv[3]) : 70, pr = argc > 4 ? atoi(argv[4]) : 70;
    const long n = long(cases.size());

    dut = new Vtm_pl;
    dut->pl_resetn = 0;
    for (int i = 0; i < 4; i++) tick();
    dut->pl_resetn = 1;
    tick();

    long sent = 0, got = 0, bad = 0, cycles = 0;
    while (got < n && cycles < 20 * n + 1000) {
        // drive this cycle's inputs, then sample the handshakes before the edge
        bool v = sent < n && int(rng() % 100) < pv;
        dut->s_axis_tvalid = v;
        if (v) {
            const Case &x = cases[sent];
            uint32_t w[8] = {x.a[0] | x.a[1] << 16, x.a[2] | x.a[3] << 16,
                             x.b[0] | x.b[1] << 16, x.b[2] | x.b[3] << 16, x.c, 0, 0, 0};
            for (int i = 0; i < 8; i++) dut->s_axis_tdata[i] = w[i];
            dut->s_axis_tlast = sent == n - 1;
        }
        dut->m_axis_tready = int(rng() % 100) < pr;
        dut->eval();
        bool in = dut->s_axis_tvalid && dut->s_axis_tready;
        if (dut->m_axis_tvalid && dut->m_axis_tready) {
            const Case &x = cases[got];
            uint32_t d0 = dut->m_axis_tdata[0], d1 = dut->m_axis_tdata[1], d2 = dut->m_axis_tdata[2];
            bool last = dut->m_axis_tlast;
            if (d0 != x.d[0] || d1 != x.d[1] || d2 != x.d[2] || dut->m_axis_tdata[3] != 0 ||
                last != (got == n - 1)) {
                if (bad < 10)
                    fprintf(stderr, "case %ld: expected %08x %08x %04x last %d, got %08x %08x %08x last %d\n", got,
                            x.d[0], x.d[1], x.d[2], got == n - 1, d0, d1, d2, last);
                bad++;
            }
            got++;
        }
        tick();
        if (in) sent++;
        cycles++;
    }
    dut->s_axis_tvalid = 0;
    dut->m_axis_tready = 1;
    for (int i = 0; i < 20; i++) tick();
    if (dut->m_axis_tvalid) bad++, fprintf(stderr, "an extra result came out\n");

    uint32_t id = lite_read(0x000), lat = lite_read(0x00C), cin = lite_read(0x014), cout = lite_read(0x018);
    uint32_t stalls = lite_read(0x01C);
    lite_write(0x020, 1);
    uint32_t cin2 = lite_read(0x014);
    bool regs_ok = id == 0x544D4952 && lat == 7 && cin == uint32_t(n) && cout == uint32_t(n) && cin2 == 0;
    printf("{\"cases\": %ld, \"received\": %ld, \"mismatches\": %ld, \"cycles\": %ld, \"stalls\": %u, "
           "\"regs_ok\": %s}\n", n, got, bad, cycles, stalls, regs_ok ? "true" : "false");
    delete dut;
    return bad || got != n || !regs_ok ? 1 : 0;
}
