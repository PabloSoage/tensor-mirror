// Verilator testbench for rtl/tm_pl_x.v: the cases of tools/rtl_vectors.py or
// tools/bs_rtl_vectors.py (the format of tb/tb_tfdpa.cpp) go in on s_axis, NB beats each,
// with random gaps; m_axis is drained with random back-pressure, and every result must
// come out once, in order, with the padding zero and tlast on the last case. Then the
// AXI4-Lite registers are read (ID, VERSION, LATENCY, L, NB, ST, counters) and the
// counters cleared.
//   tb <vectors.txt> [seed] [p(valid) percent] [p(ready) percent]
// Build with -DTB_L, -DTB_AW, -DTB_BW, -DTB_ST as for tb_tfdpa.cpp, and -DTB_LAT (tfdpa's
// LATENCY). Exit status 1 on any mismatch.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "Vtm_pl_x.h"
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
#ifndef TB_ST
#define TB_ST 0
#endif
#ifndef TB_LAT
#define TB_LAT 8
#endif
constexpr int NB = (TB_L * TB_AW + TB_L * TB_BW + 48 + 255) / 256;

struct Case {
    std::vector<uint32_t> w;  // the NB beats, 8 words each
    unsigned d;
};

static Vtm_pl_x *dut;
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

// Appends the low n bits of x at bit position *pos of w.
static void put(std::vector<uint32_t> &w, int *pos, uint32_t x, int n) {
    for (int i = 0; i < n; i++, (*pos)++)
        if (x >> i & 1) w[*pos / 32] |= 1u << (*pos % 32);
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
    for (;;) {
        unsigned a[TB_L], b[TB_L], c, sfa = 0, sfb = 0, d;
        int got = 0;
        for (int k = 0; k < TB_L; k++) got += fscanf(f, "%x", &a[k]);
        for (int k = 0; k < TB_L; k++) got += fscanf(f, "%x", &b[k]);
        got += fscanf(f, "%x", &c);
        if (TB_ST) got += fscanf(f, "%x %x", &sfa, &sfb);
        got += fscanf(f, "%x", &d);
        if (got != 2 * TB_L + 2 + 2 * TB_ST) break;
        Case x{std::vector<uint32_t>(8 * NB, 0), d};
        int pos = 0;
        for (int k = 0; k < TB_L; k++) put(x.w, &pos, a[k], TB_AW);
        for (int k = 0; k < TB_L; k++) put(x.w, &pos, b[k], TB_BW);
        put(x.w, &pos, c, 32);
        put(x.w, &pos, sfa, 8);
        put(x.w, &pos, sfb, 8);
        cases.push_back(x);
    }
    fclose(f);
    std::mt19937 rng(argc > 2 ? atoi(argv[2]) : 1);
    const int pv = argc > 3 ? atoi(argv[3]) : 70, pr = argc > 4 ? atoi(argv[4]) : 70;
    const long n = long(cases.size());

    dut = new Vtm_pl_x;
    dut->pl_resetn = 0;
    for (int i = 0; i < 4; i++) tick();
    dut->pl_resetn = 1;
    tick();

    long sent = 0, got = 0, bad = 0, cycles = 0;  // sent counts beats
    while (got < n && cycles < 20 * NB * n + 1000) {
        // drive this cycle's inputs, then sample the handshakes before the edge
        bool v = sent < NB * n && int(rng() % 100) < pv;
        dut->s_axis_tvalid = v;
        if (v) {
            const Case &x = cases[sent / NB];
            for (int i = 0; i < 8; i++) dut->s_axis_tdata[i] = x.w[8 * (sent % NB) + i];
            dut->s_axis_tlast = sent == NB * n - 1;
        }
        dut->m_axis_tready = int(rng() % 100) < pr;
        dut->eval();
        bool in = dut->s_axis_tvalid && dut->s_axis_tready;
        if (dut->m_axis_tvalid && dut->m_axis_tready) {
            const Case &x = cases[got];
            uint32_t d = dut->m_axis_tdata[0];
            bool pad = dut->m_axis_tdata[1] | dut->m_axis_tdata[2] | dut->m_axis_tdata[3];
            bool last = dut->m_axis_tlast;
            if (d != x.d || pad || last != (got == n - 1)) {
                if (bad < 10)
                    fprintf(stderr, "case %ld: expected %08x last %d, got %08x pad %d last %d\n", got, x.d,
                            got == n - 1, d, pad, last);
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
    for (int i = 0; i < 20 + TB_LAT; i++) tick();
    if (dut->m_axis_tvalid) bad++, fprintf(stderr, "an extra result came out\n");

    uint32_t id = lite_read(0x000), ver = lite_read(0x004), lat = lite_read(0x00C);
    uint32_t cin = lite_read(0x014), cout = lite_read(0x018), stalls = lite_read(0x01C);
    uint32_t l = lite_read(0x024), nb = lite_read(0x028), st = lite_read(0x02C);
    lite_write(0x020, 1);
    uint32_t cin2 = lite_read(0x014);
    bool regs_ok = id == 0x544D4952 && ver == 2 && lat == TB_LAT && cin == uint32_t(NB * n) &&
                   cout == uint32_t(n) && l == TB_L && nb == NB && st == TB_ST && cin2 == 0;
    printf("{\"cases\": %ld, \"beats_per_case\": %d, \"received\": %ld, \"mismatches\": %ld, \"cycles\": %ld, "
           "\"stalls\": %u, \"regs_ok\": %s}\n", n, NB, got, bad, cycles, stalls, regs_ok ? "true" : "false");
    delete dut;
    return bad || got != n || !regs_ok ? 1 : 0;
}
