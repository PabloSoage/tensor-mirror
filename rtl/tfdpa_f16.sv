// Truncated fused dot-product-add with f16 operands: d = c + sum(a[k] * b[k]), as Volta's
// HMMA computes it (MMA-Sim's T-FDPA, arXiv 2511.10909, Algorithm 7; Volta: L = 4, F = 23).
// rtl/tfdpa.sv with Volta's formats; this module keeps the interface that rtl/tm_pl.v and the
// Volta testbenches use. Up to a010ee3 it was a datapath of its own, with the same 8 stages
// (the KV260 bitstreams 35ffa34 and 3e535a9 were built from that).
//
// C_F16 = 0: c is f32 (in c[31:0]); 1: c is f16 (in c[15:0], c[31:16] ignored).
// D_F16 = 0: d is f32, rounded toward zero; 1: d is f16 (in d[15:0], d[31:16] = 0), rounded
// to nearest even. Volta's three mma.m8n8k4 variants are (C_F16, D_F16) = (0, 0), (1, 0)
// and (1, 1).
//
// PIPE = 0: combinational (clk unused). PIPE = 1: one result per cycle while ce is 1,
// LATENCY = 8 cycles from a, b, c to d.

module tfdpa_f16 #(
    parameter int L = 4,                             // terms per group
    parameter int F = 23,                            // fractional bits of the fused sum (>= 23)
    parameter int C_F16 = 0,
    parameter int D_F16 = 0,
    // exponent given to a zero term, as measured on the V100 (H-002): MMA-Sim has -20 for
    // C in f16; the GPU keeps every product bit with D in f32 (<= -25) and stops at 2**-42
    // with D in f16 (-19)
    parameter int EZERO = C_F16 == 0 ? -131 : D_F16 != 0 ? -19 : -25,
    parameter int PIPE = 0                           // 0 or 1
) (
    input  logic               clk,
    input  logic               ce,  // PIPE = 1: every stage holds while ce is 0
    input  logic [L-1:0][15:0] a,
    input  logic [L-1:0][15:0] b,
    input  logic [31:0]        c,
    output logic [31:0]        d
);
    /* verilator lint_off UNUSEDPARAM */
    localparam int LATENCY = PIPE != 0 ? 8 : 0;  // for instantiating code and the testbench
    /* verilator lint_on UNUSEDPARAM */

    tfdpa #(
        .L(L), .F(F), .EZERO(EZERO), .RNE(D_F16 != 0 ? 1 : 0),
        .AW(16), .AE(5), .AM(10), .ALSB(0), .ASPEC(0),
        .BW(16), .BE(5), .BM(10), .BLSB(0), .BSPEC(0),
        .CE(C_F16 != 0 ? 5 : 8), .CM(C_F16 != 0 ? 10 : 23), .CLSB(0),
        .DE(D_F16 != 0 ? 5 : 8), .DM(D_F16 != 0 ? 10 : 23), .DLSB(0),
        .ST(0), .XP(0), .PIPE(PIPE)
    ) u (
        .clk(clk), .ce(ce), .a(a), .b(b), .c(c), .sfa(8'd0), .sfb(8'd0), .d(d)
    );
endmodule
