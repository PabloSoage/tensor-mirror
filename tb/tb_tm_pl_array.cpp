// Verilator testbench for rtl/tm_pl_array.v: the input beats of tools/gemm_cases.py rtl
// (jobs x NIN beats of 32 bytes) go in on s_axis with random gaps; m_axis is drained with
// random back-pressure, every output beat is written to <out.bin> and compared with
// <expected.bin> (jobs x NOUT beats of 16 bytes), with tlast on each job's last beat. Then
// the AXI4-Lite registers are read and the counters cleared.
//   tb <in.bin> <expected.bin> <out.bin> [seed] [p(valid) percent] [p(ready) percent]
// Build with -DTB_NIN, -DTB_NOUT (beats per job), -DTB_LAT (the unit's LATENCY), -DTB_L,
// -DTB_X1, -DTB_X2, -DTB_X3 (the registers 0x028..0x030). Exit status 1 on any mismatch.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "Vtm_pl_array.h"
#include "verilated.h"

static Vtm_pl_array *dut;
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

static std::vector<uint32_t> read_words(const char *path) {
    std::vector<uint32_t> w;
    FILE *f = fopen(path, "rb");
    if (!f) perror(path), exit(2);
    uint32_t x;
    while (fread(&x, 4, 1, f) == 1) w.push_back(x);
    fclose(f);
    return w;
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    if (argc < 4) {
        fprintf(stderr, "usage: %s <in.bin> <expected.bin> <out.bin> [seed] [p_valid%%] [p_ready%%]\n", argv[0]);
        return 2;
    }
    const std::vector<uint32_t> in = read_words(argv[1]), exp = read_words(argv[2]);
    const long jobs = long(in.size()) / (8 * TB_NIN);
    if (long(in.size()) != jobs * 8 * TB_NIN || long(exp.size()) != jobs * 4 * TB_NOUT) {
        fprintf(stderr, "file sizes do not match %d input and %d output beats per job\n", TB_NIN, TB_NOUT);
        return 2;
    }
    FILE *fo = fopen(argv[3], "wb");
    if (!fo) return perror(argv[3]), 2;
    std::mt19937 rng(argc > 4 ? atoi(argv[4]) : 1);
    const int pv = argc > 5 ? atoi(argv[5]) : 70, pr = argc > 6 ? atoi(argv[6]) : 70;
    const long nin = jobs * TB_NIN, nout = jobs * TB_NOUT;

    dut = new Vtm_pl_array;
    dut->pl_resetn = 0;
    for (int i = 0; i < 4; i++) tick();
    dut->pl_resetn = 1;
    tick();

    long sent = 0, got = 0, bad = 0, cycles = 0, limit = 40 * (nin + nout) + 20000 * jobs;
    while (got < nout && cycles < limit) {
        bool v = sent < nin && int(rng() % 100) < pv;
        dut->s_axis_tvalid = v;
        if (v) {
            for (int i = 0; i < 8; i++) dut->s_axis_tdata[i] = in[8 * sent + i];
            dut->s_axis_tlast = (sent + 1) % TB_NIN == 0;
        }
        dut->m_axis_tready = int(rng() % 100) < pr;
        dut->eval();
        bool took = dut->s_axis_tvalid && dut->s_axis_tready;
        if (dut->m_axis_tvalid && dut->m_axis_tready) {
            bool diff = false;
            for (int i = 0; i < 4; i++) {
                uint32_t w = dut->m_axis_tdata[i];
                fwrite(&w, 4, 1, fo);
                diff |= w != exp[4 * got + i];
            }
            bool last = dut->m_axis_tlast;
            if (diff || last != ((got + 1) % TB_NOUT == 0)) {
                if (bad < 10)
                    fprintf(stderr, "job %ld beat %ld: %08x %08x %08x %08x last %d, expected %08x %08x %08x %08x\n",
                            got / TB_NOUT, got % TB_NOUT, dut->m_axis_tdata[0], dut->m_axis_tdata[1],
                            dut->m_axis_tdata[2], dut->m_axis_tdata[3], last, exp[4 * got], exp[4 * got + 1],
                            exp[4 * got + 2], exp[4 * got + 3]);
                bad++;
            }
            got++;
        }
        tick();
        if (took) sent++;
        cycles++;
    }
    fclose(fo);
    dut->s_axis_tvalid = 0;
    dut->m_axis_tready = 1;
    for (int i = 0; i < 20000; i++) tick();
    if (dut->m_axis_tvalid) bad++, fprintf(stderr, "an extra beat came out\n");

    uint32_t id = lite_read(0x000), ver = lite_read(0x004), lat = lite_read(0x00C);
    uint32_t cin = lite_read(0x014), cout = lite_read(0x018), stalls = lite_read(0x01C);
    uint32_t x0 = lite_read(0x024), x1 = lite_read(0x028), x2 = lite_read(0x02C), x3 = lite_read(0x030);
    lite_write(0x020, 1);
    uint32_t cin2 = lite_read(0x014);
    bool regs_ok = id == 0x544D4952 && ver == 3 && lat == TB_LAT && cin == uint32_t(nin) &&
                   cout == uint32_t(nout) && x0 == TB_L && x1 == TB_X1 && x2 == TB_X2 && x3 == TB_X3 &&
                   cin2 == 0;
    printf("{\"jobs\": %ld, \"beats_in\": %ld, \"beats_out\": %ld, \"mismatched_beats\": %ld, \"cycles\": %ld, "
           "\"cycles_per_job\": %.1f, \"stalls\": %u, \"regs_ok\": %s}\n",
           jobs, sent, got, bad, cycles, double(cycles) / (jobs ? jobs : 1), stalls, regs_ok ? "true" : "false");
    delete dut;
    return bad || got != nout || !regs_ok ? 1 : 0;
}
