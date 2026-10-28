// tm_array - R x C rtl/tfdpa.sv units, output-stationary: unit (r, c) chains d = c + a . b
// over the steps of K, groups of L, each group's d (in D's format) the c of the next, as
// python/tensormirror/tfdpa.py dot does and as a loop of mma.sync does on the GPU (H-024,
// H-027). Row r of units shares a, column c shares b.
//
// A unit's d comes LATENCY cycles after its inputs, so each unit carries Q chains at once,
// one per cycle ("slots"): a slot's next step goes in Q cycles after its last one, when
// that d, delayed Q - LATENCY cycles more, is on the unit's c. The caller feeds a block as
// S steps x Q slots, one slot per cycle without gaps, in_first on the first step (c from
// the c port) and in_last on the last; out_v then marks, LATENCY cycles later, the cycles
// whose d is a slot's final result. Between blocks the input may pause.

module tm_array #(
    parameter int R = 4,        // rows of units
    parameter int C = 4,        // columns of units
    parameter int Q = 16,       // slots per unit, >= the unit's LATENCY
    parameter int L = 4,
    parameter int F = 23,
    parameter int EZERO = -131,
    parameter int RNE = 0,
    parameter int AW = 16, parameter int AE = 5, parameter int AM = 10,
    parameter int ALSB = 0, parameter int ASPEC = 0,
    parameter int BW = 16, parameter int BE = 5, parameter int BM = 10,
    parameter int BLSB = 0, parameter int BSPEC = 0,
    parameter int CE = 8, parameter int CM = 23, parameter int CLSB = 0,
    parameter int DE = 8, parameter int DM = 23, parameter int DLSB = 0,
    parameter int XP = 0,
    parameter int SP = 0
) (
    input  logic                            clk,
    input  logic                            in_v,
    input  logic                            in_first,
    input  logic                            in_last,
    input  logic [R-1:0][L*AW-1:0]          a,  // row r: a[k] in bits AW*k+AW-1..AW*k
    input  logic [C-1:0][L*BW-1:0]          b,
    input  logic [R-1:0][C-1:0][31:0]       c,  // read on in_first
    output logic                            out_v,
    output logic [R-1:0][C-1:0][31:0]       d
);
    localparam int LATENCY = 8 + (XP != 0 ? 2 * ($clog2(L + 1) / 2) : 0) + (SP != 0 ? 1 : 0);
`ifdef MUT_SLOT  // mutant for the testbench: the feedback is one cycle late, another slot's d
    localparam int FB = Q - LATENCY + 1;
`else
    localparam int FB = Q - LATENCY;
`endif
    if (Q < LATENCY) begin : g_bad_q
        $error("tm_array: Q must be at least the unit's LATENCY");
    end

    for (genvar r = 0; r < R; r++) begin : g_row
        for (genvar cc = 0; cc < C; cc++) begin : g_col
            logic [31:0] fb;  // this slot's d from its last step
            pipe_delay #(.W(32), .D(FB)) u_fb (.clk(clk), .ce(1'b1), .x(d[r][cc]), .y(fb));
            tfdpa #(
                .L(L), .F(F), .EZERO(EZERO), .RNE(RNE),
                .AW(AW), .AE(AE), .AM(AM), .ALSB(ALSB), .ASPEC(ASPEC),
                .BW(BW), .BE(BE), .BM(BM), .BLSB(BLSB), .BSPEC(BSPEC),
                .CE(CE), .CM(CM), .CLSB(CLSB), .DE(DE), .DM(DM), .DLSB(DLSB),
                .ST(0), .XP(XP), .SP(SP), .PIPE(1)
            ) u (
                .clk(clk), .ce(1'b1), .a(a[r]), .b(b[cc]), .c(in_first ? c[r][cc] : fb),
                .sfa(8'd0), .sfb(8'd0), .d(d[r][cc]));
        end
    end

    pipe_delay #(.W(1), .D(LATENCY)) u_v (.clk(clk), .ce(1'b1), .x(in_v && in_last), .y(out_v));
endmodule
