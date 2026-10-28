// tm_pl_array - a PL top for the KV260 build with an R x C rtl/tm_array.sv of one
// instruction (tfdpa's parameters, from tools/rtl_params.py), computing one GEMM job of a
// fixed size per build, D = C + A B with A MJ x KJ, B KJ x NJ, C and D MJ x NJ, fed and
// drained by the AXI DMA of rtl/tm_pl.v's system. K is chained in order, groups of L,
// through D's format: a loop of mma.sync on the GPU (H-027).
//
//   s_axis  a job is A, then B^T, then C, each row major and packed without gaps (element
//           k of a row in the bits AW*k.., the first element of a beat in its lowest bits):
//           MJ*KJ*AW/256 beats of A, NJ*KJ*BW/256 of B^T, MJ*NJ*EW/256 of C, with EW the
//           width of C's and D's elements (16 or 32; D's format is C's).
//   m_axis  D, row major, 128/EW elements per beat, MJ*NJ*EW/128 beats per job; tlast on
//           the last beat of a job whose last input beat had tlast, so one DMA transfer can
//           hold many jobs. A job's D comes out after its whole input, and the next job is
//           taken after it (no double buffering).
//   s_axi   AXI4-Lite registers (rtl/tm_regs.v), VERSION 3, LATENCY the unit's, with
//             0x024 L   0x028 Q * 65536 + R * 256 + C
//             0x02C s_axis beats per job   0x030 m_axis beats per job
//
// Inside, A is in R banks (row i in bank i % R) and B^T in C banks (row j in bank j % C),
// 256-bit words; C and D share R x C banks, element (i, j) in bank (i % R, j % C), WE =
// 256 / EW / C elements per word. The array computes blocks of R*Q x C outputs: slot q of
// unit (r, c) is row mb*R*Q + q*R + r, column nb*C + c. Every dimension is a power of 2.

module tm_pl_array #(
    parameter [31:0] BUILD = 32'h0,
    parameter R = 4, parameter C = 4, parameter Q = 16,
    parameter MJ = 64, parameter NJ = 64, parameter KJ = 256,
    parameter L = 4,
    parameter F = 23,
    parameter EZERO = -131,
    parameter RNE = 0,
    parameter AW = 16, parameter AE = 5, parameter AM = 10, parameter ALSB = 0, parameter ASPEC = 0,
    parameter BW = 16, parameter BE = 5, parameter BM = 10, parameter BLSB = 0, parameter BSPEC = 0,
    parameter CE = 8, parameter CM = 23, parameter CLSB = 0,
    parameter DE = 8, parameter DM = 23, parameter DLSB = 0,
    parameter XP = 0,
    parameter SP = 0
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axi:s_axis:m_axis, ASSOCIATED_RESET pl_resetn" *)
    input  wire         clk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 pl_resetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire         pl_resetn,
    input  wire [11:0]  s_axi_awaddr,
    input  wire         s_axi_awvalid,
    output wire         s_axi_awready,
    input  wire [31:0]  s_axi_wdata,
    input  wire [3:0]   s_axi_wstrb,
    input  wire         s_axi_wvalid,
    output wire         s_axi_wready,
    output wire [1:0]   s_axi_bresp,
    output wire         s_axi_bvalid,
    input  wire         s_axi_bready,
    input  wire [11:0]  s_axi_araddr,
    input  wire         s_axi_arvalid,
    output wire         s_axi_arready,
    output wire [31:0]  s_axi_rdata,
    output wire [1:0]   s_axi_rresp,
    output wire         s_axi_rvalid,
    input  wire         s_axi_rready,
    input  wire [255:0] s_axis_tdata,
    input  wire         s_axis_tvalid,
    output wire         s_axis_tready,
    input  wire         s_axis_tlast,
    output wire [127:0] m_axis_tdata,
    output wire [15:0]  m_axis_tkeep,
    output wire         m_axis_tvalid,
    input  wire         m_axis_tready,
    output wire         m_axis_tlast
);
    localparam LATENCY = 8 + (XP != 0 ? 2 * ($clog2(L + 1) / 2) : 0) + (SP != 0 ? 1 : 0);
    localparam EW = DLSB + DE + DM + 1;     // bits of a C or D element
    localparam BPRA = KJ * AW / 256;        // beats per row of A
    localparam BPRB = KJ * BW / 256;        // beats per row of B^T
    localparam WPA = 256 / (L * AW);        // groups of L per word of A
    localparam WPB = 256 / (L * BW);
    localparam S = KJ / L;                  // steps of a chain
    localparam WE = 256 / EW / C;           // elements per word of a C/D bank
    localparam BPRC = NJ * EW / 256;        // beats per row of C, words per row of a C/D bank
    localparam ME = 128 / EW;               // elements per beat of D
    localparam MEC = ME / C;                // of them, per bank
    localparam BPRD = NJ * EW / 128;        // beats per row of D
    localparam MB = MJ / (R * Q);           // blocks down M
    localparam NBK = NJ / C;                // blocks across N
    localparam NIA = MJ * BPRA, NIB = NJ * BPRB, NIC = MJ * BPRC;  // input beats
    localparam NIN = NIA + NIB + NIC;
    localparam NOUT = MJ * BPRD;            // output beats
    localparam DA = MJ / R * BPRA, DB = NJ / C * BPRB, DC = MJ / R * BPRC;  // bank depths

    localparam LOAD = 2'd0, RUN = 2'd1, WAIT = 2'd2, DRAIN = 2'd3;
    reg [1:0] state;

    // ---- load: the job's beats into the banks ----------------------------------------
`ifdef MUT_ALWAYS_READY  // mutant for the testbench: beats sent while the array runs are lost
    assign s_axis_tready = 1'b1;
`else
    assign s_axis_tready = state == LOAD;
`endif
    wire beat = s_axis_tvalid && s_axis_tready;
    integer n;  // input beats of this job so far
    wire in_a = n < NIA, in_b = !in_a && n < NIA + NIB, in_c = !in_a && !in_b;
    wire [31:0] nb_ = n - NIA, nc_ = n - NIA - NIB;
    wire [31:0] ia = n / BPRA, ib = nb_ / BPRB, ic = nc_ / BPRC;  // the row being loaded
    wire [31:0] wa = (ia / R) * BPRA + n % BPRA;
    wire [31:0] wb = (ib / C) * BPRB + nb_ % BPRB;
    wire [31:0] wc = (ic / R) * BPRC + nc_ % BPRC;

    // ---- run: issue counters (stage I), bank reads, then the array (stage R) -----------
    integer mb, nbk, s, q, wait_n;
    wire issue = state == RUN;
    wire [31:0] ii = mb * Q + q;            // row of every slot-q unit, divided by R
    wire [31:0] ra = ii * BPRA + s / WPA;
    wire [31:0] rb = nbk * BPRB + s / WPB;
    wire [31:0] rc = ii * BPRC + nbk / WE;
    reg v1, first1, last1;
    reg [31:0] sa1, sb1, lane1, rc1;
    always @(posedge clk) begin
        v1 <= issue;
        first1 <= s == 0;
        last1 <= s == S - 1;
        sa1 <= s % WPA;
        sb1 <= s % WPB;
        lane1 <= nbk % WE;
        rc1 <= rc;
    end

    // ---- drain: D out of the C/D banks, through a 2-entry FIFO ------------------------
    integer o;                              // output beats read so far
    reg [1:0] cnt;                          // entries in the FIFO
    reg inflight;                           // a read whose data reaches the FIFO next cycle
    wire pop = m_axis_tvalid && m_axis_tready;
    wire rd_d = state == DRAIN && o < NOUT && cnt + inflight - pop <= 1;
    wire [31:0] io = o / BPRD, j0 = (o % BPRD) * ME;
    wire [31:0] rd_addr = (io / R) * BPRC + (j0 / C) / WE;
    reg [31:0] dr1, dlane1;
    reg dlast1;
    reg job_last;                           // this job's last input beat had tlast
    always @(posedge clk) begin
        dr1 <= io % R;
        dlane1 <= (j0 / C) % WE;
        dlast1 <= o == NOUT - 1 && job_last;
        if (beat && n == NIN - 1) job_last <= s_axis_tlast;
    end

    // ---- the banks ----------------------------------------------------------------------
    wire [R*256-1:0] a_rd;
    wire [C*256-1:0] b_rd;
    wire [R*C*WE*EW-1:0] cd_rd;
    wire [R*C*32-1:0] d_arr;
    wire out_v;
    wire [31:0] res_addr, res_lane;         // where a final d goes, LATENCY after stage R
    pipe_delay #(.W(32), .D(LATENCY)) u_ra (.clk(clk), .ce(1'b1), .x(rc1), .y(res_addr));
    pipe_delay #(.W(32), .D(LATENCY)) u_rl (.clk(clk), .ce(1'b1), .x(lane1), .y(res_lane));
    genvar r, c, v;
    generate
        for (r = 0; r < R; r = r + 1) begin : g_a
            reg [255:0] mem [0:DA-1];
            reg [255:0] rd;
            always @(posedge clk) begin
                if (beat && in_a && ia % R == r) mem[wa] <= s_axis_tdata;
                rd <= mem[ra];
            end
            assign a_rd[r*256 +: 256] = rd;
        end
        for (c = 0; c < C; c = c + 1) begin : g_b
            reg [255:0] mem [0:DB-1];
            reg [255:0] rd;
            always @(posedge clk) begin
                if (beat && in_b && ib % C == c) mem[wb] <= s_axis_tdata;
                rd <= mem[rb];
            end
            assign b_rd[c*256 +: 256] = rd;
        end
        for (r = 0; r < R; r = r + 1) begin : g_cr
            for (c = 0; c < C; c = c + 1) begin : g_cc
                // element v of a word is column c + C * v of its group of 256 / EW
                wire [WE*EW-1:0] load_word;
                for (v = 0; v < WE; v = v + 1) begin : g_v
                    assign load_word[v*EW +: EW] = s_axis_tdata[(c + C*v)*EW +: EW];
                end
                wire load = beat && in_c && ic % R == r;
                wire res = out_v;
                wire [31:0] waddr = load ? wc : res_addr;
                wire [WE*EW-1:0] wdata = load ? load_word : {WE{d_arr[(r*C + c)*32 +: EW]}};
                wire [31:0] raddr = state == DRAIN ? rd_addr : rc;
                reg [WE*EW-1:0] mem [0:DC-1];
                reg [WE*EW-1:0] rd;
                integer k;
                always @(posedge clk) begin
                    for (k = 0; k < WE; k = k + 1)
                        if (load || (res && res_lane == k))
                            mem[waddr][k*EW +: EW] <= wdata[k*EW +: EW];
                    rd <= mem[raddr];
                end
                assign cd_rd[(r*C + c)*WE*EW +: WE*EW] = rd;
            end
        end
    endgenerate

    // ---- the array ------------------------------------------------------------------------
    wire [R*L*AW-1:0] a_in;
    wire [C*L*BW-1:0] b_in;
    wire [R*C*32-1:0] c_in;
    generate
        for (r = 0; r < R; r = r + 1) begin : g_ain
            assign a_in[r*L*AW +: L*AW] = a_rd[r*256 + sa1*L*AW +: L*AW];
        end
        for (c = 0; c < C; c = c + 1) begin : g_bin
            assign b_in[c*L*BW +: L*BW] = b_rd[c*256 + sb1*L*BW +: L*BW];
        end
        for (r = 0; r < R; r = r + 1) begin : g_cin
            for (c = 0; c < C; c = c + 1) begin : g_cinc
                assign c_in[(r*C + c)*32 +: 32] =
                    {{(32 - EW){1'b0}}, cd_rd[(r*C + c)*WE*EW + lane1*EW +: EW]};
            end
        end
    endgenerate
    tm_array #(
        .R(R), .C(C), .Q(Q), .L(L), .F(F), .EZERO(EZERO), .RNE(RNE),
        .AW(AW), .AE(AE), .AM(AM), .ALSB(ALSB), .ASPEC(ASPEC),
        .BW(BW), .BE(BE), .BM(BM), .BLSB(BLSB), .BSPEC(BSPEC),
        .CE(CE), .CM(CM), .CLSB(CLSB), .DE(DE), .DM(DM), .DLSB(DLSB), .XP(XP), .SP(SP)
    ) u_array (
        .clk(clk), .in_v(v1), .in_first(first1), .in_last(last1), .a(a_in), .b(b_in), .c(c_in),
        .out_v(out_v), .d(d_arr));

    // ---- the output FIFO ----------------------------------------------------------------
    reg [128:0] fifo [0:1];                 // {last, data}, entry 0 the head
    reg [127:0] dword1;                     // the beat read last cycle
    integer cc, vv;
    always @* begin
        dword1 = 128'd0;
        for (cc = 0; cc < C; cc = cc + 1)
            for (vv = 0; vv < MEC; vv = vv + 1)
                dword1[(cc + C*vv)*EW +: EW] = cd_rd[(dr1*C + cc)*WE*EW + (dlane1 + vv)*EW +: EW];
    end
    assign m_axis_tvalid = cnt != 0;
    assign m_axis_tdata = fifo[0][127:0];
    assign m_axis_tlast = fifo[0][128];
    assign m_axis_tkeep = 16'hffff;

    // ---- control ----------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!pl_resetn) begin
            state <= LOAD;
            n <= 0;
            mb <= 0; nbk <= 0; s <= 0; q <= 0; wait_n <= 0;
            o <= 0;
            cnt <= 0;
            inflight <= 0;
        end else begin
            case (state)
                LOAD: if (beat) begin
                    if (n == NIN - 1) begin
                        n <= 0;
                        state <= RUN;
                    end else
                        n <= n + 1;
                end
                RUN: begin
                    q <= q == Q - 1 ? 0 : q + 1;
                    if (q == Q - 1) begin
                        s <= s == S - 1 ? 0 : s + 1;
                        if (s == S - 1) begin
                            nbk <= nbk == NBK - 1 ? 0 : nbk + 1;
                            if (nbk == NBK - 1) begin
                                mb <= mb == MB - 1 ? 0 : mb + 1;
                                if (mb == MB - 1) state <= WAIT;
                            end
                        end
                    end
                end
                WAIT: begin  // the last results reach the banks
                    wait_n <= wait_n + 1;
                    if (wait_n == LATENCY + 2) begin
                        wait_n <= 0;
                        state <= DRAIN;
                    end
                end
                DRAIN: if (o == NOUT && cnt == 0 && !inflight) begin
                    o <= 0;
                    state <= LOAD;
                end
                default: state <= LOAD;
            endcase
            if (rd_d) o <= o + 1;
            inflight <= rd_d;
            // the FIFO: pop the head, push the beat read last cycle
            if (pop) fifo[0] <= fifo[1];
            if (inflight) fifo[cnt == 2 || (cnt == 1 && !pop)] <= {dlast1, dword1};  // at cnt - pop
            cnt <= cnt + inflight - pop;
        end
    end

    // ---- registers and counters (rtl/tm_regs.v) -----------------------------------------------
    tm_regs #(.VERSION(3), .BUILD(BUILD), .LATENCY(LATENCY), .F(F), .X0(L), .X1(Q * 65536 + R * 256 + C),
              .X2(NIN), .X3(NOUT)) u_regs (
        .clk(clk), .pl_resetn(pl_resetn),
        .s_axi_awaddr(s_axi_awaddr), .s_axi_awvalid(s_axi_awvalid), .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata), .s_axi_wstrb(s_axi_wstrb), .s_axi_wvalid(s_axi_wvalid),
        .s_axi_wready(s_axi_wready), .s_axi_bresp(s_axi_bresp), .s_axi_bvalid(s_axi_bvalid),
        .s_axi_bready(s_axi_bready), .s_axi_araddr(s_axi_araddr), .s_axi_arvalid(s_axi_arvalid),
        .s_axi_arready(s_axi_arready), .s_axi_rdata(s_axi_rdata), .s_axi_rresp(s_axi_rresp),
        .s_axi_rvalid(s_axi_rvalid), .s_axi_rready(s_axi_rready),
        .beat_in(beat), .beat_out(pop), .stall(m_axis_tvalid && !m_axis_tready));
endmodule
