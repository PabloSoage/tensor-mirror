// tm_pl_x - a PL top for the KV260 build with one rtl/tfdpa.sv unit of any instruction
// (its parameters are tfdpa's, from tools/rtl_params.py), fed and drained by the AXI DMA
// of rtl/tm_pl.v's system.
//
//   s_axis  a case is NB beats: its bits {sfb, sfa, c, b, a}, a in the lowest, packed
//           without gaps (a[k] in bits AW*k+AW-1..AW*k, then b, then the 32 bits of c,
//           then sfa and sfb, 8 bits each) and cut into 256-bit beats, the first beat
//           holding the lowest bits. NB = ceil((L*AW + L*BW + 48) / 256).
//   m_axis  one result per beat, to the DMA's S2MM, LATENCY beats after its case's last
//             tdata[31:0]    d
//             tdata[127:32]  zero
//           tlast follows the tlast of the case's last beat.
//   s_axi   AXI4-Lite registers (rtl/tm_regs.v), VERSION 2, with
//             0x024 L   0x028 NB   0x02C ST
//
// The unit (PIPE = 1) moves when the output register is empty or being taken, and s_axis
// is ready exactly then, as in tm_pl. A case's first NB - 1 beats are held here; its last
// beat goes to the unit with them, so the unit takes one case every NB beats.

module tm_pl_x #(
    parameter [31:0] BUILD = 32'h0,
    parameter L = 4,
    parameter F = 23,
    parameter EZERO = -131,
    parameter RNE = 0,
    parameter AW = 16, parameter AE = 5, parameter AM = 10, parameter ALSB = 0, parameter ASPEC = 0,
    parameter BW = 16, parameter BE = 5, parameter BM = 10, parameter BLSB = 0, parameter BSPEC = 0,
    parameter CE = 8, parameter CM = 23, parameter CLSB = 0,
    parameter DE = 8, parameter DM = 23, parameter DLSB = 0,
    parameter ST = 0,
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
    // tfdpa's LATENCY with PIPE = 1
    localparam LATENCY = 8 + (XP != 0 ? 2 * ($clog2(L + 1) / 2) : 0) + (SP != 0 ? 1 : 0);
    localparam NB = (L * AW + L * BW + 48 + 255) / 256;  // s_axis beats per case

    // ---- the datapath ----------------------------------------------------------------
    wire ce = !m_axis_tvalid || m_axis_tready;
`ifdef MUT_ALWAYS_READY  // mutant for the testbench: ignores back-pressure, so cases are lost
    assign s_axis_tready = 1'b1;
`else
    assign s_axis_tready = ce;
`endif
    wire beat = s_axis_tvalid && s_axis_tready;

    // a case's bits: the beats held so far and the one on s_axis
    wire [NB*256-1:0] word;
    wire fire;  // the last beat of a case is taken
    generate
        if (NB == 1) begin : g_one
            assign word = s_axis_tdata;
            assign fire = beat;
        end else begin : g_gather
            reg [(NB-1)*256-1:0] held;
            integer n;  // beats of this case held
            always @(posedge clk)
                if (!pl_resetn)
                    n <= 0;
                else if (beat) begin
                    if (n == NB - 1) begin
                        n <= 0;
                    end else begin
                        held[n*256 +: 256] <= s_axis_tdata;
                        n <= n + 1;
                    end
                end
            assign word = {s_axis_tdata, held};
            assign fire = beat && n == NB - 1;
        end
    endgenerate

    wire [31:0] d;
    tfdpa #(
        .L(L), .F(F), .EZERO(EZERO), .RNE(RNE),
        .AW(AW), .AE(AE), .AM(AM), .ALSB(ALSB), .ASPEC(ASPEC),
        .BW(BW), .BE(BE), .BM(BM), .BLSB(BLSB), .BSPEC(BSPEC),
        .CE(CE), .CM(CM), .CLSB(CLSB), .DE(DE), .DM(DM), .DLSB(DLSB),
        .ST(ST), .XP(XP), .SP(SP), .PIPE(1)
    ) u (
        .clk(clk), .ce(ce), .a(word[L*AW-1:0]), .b(word[L*AW +: L*BW]),
        .c(word[L*AW+L*BW +: 32]), .sfa(word[L*AW+L*BW+32 +: 8]), .sfb(word[L*AW+L*BW+40 +: 8]),
        .d(d));

    // valid and last travel beside the data, moving with the same enable
    reg [LATENCY-1:0] v, last;
    always @(posedge clk) begin
        if (!pl_resetn) begin
            v <= 0;
            last <= 0;
        end else if (ce) begin
            v <= {v[LATENCY-2:0], fire};
            last <= {last[LATENCY-2:0], fire && s_axis_tlast};
        end
    end
    assign m_axis_tvalid = v[LATENCY-1];
    assign m_axis_tlast = last[LATENCY-1];
    assign m_axis_tdata = {96'd0, d};
    assign m_axis_tkeep = 16'hffff;

    // ---- registers and counters (rtl/tm_regs.v) --------------------------------------
    tm_regs #(.VERSION(2), .BUILD(BUILD), .LATENCY(LATENCY), .F(F), .X0(L), .X1(NB), .X2(ST)) u_regs (
        .clk(clk), .pl_resetn(pl_resetn),
        .s_axi_awaddr(s_axi_awaddr), .s_axi_awvalid(s_axi_awvalid), .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata), .s_axi_wstrb(s_axi_wstrb), .s_axi_wvalid(s_axi_wvalid),
        .s_axi_wready(s_axi_wready), .s_axi_bresp(s_axi_bresp), .s_axi_bvalid(s_axi_bvalid),
        .s_axi_bready(s_axi_bready), .s_axi_araddr(s_axi_araddr), .s_axi_arvalid(s_axi_arvalid),
        .s_axi_arready(s_axi_arready), .s_axi_rdata(s_axi_rdata), .s_axi_rresp(s_axi_rresp),
        .s_axi_rvalid(s_axi_rvalid), .s_axi_rready(s_axi_rready),
        .beat_in(beat), .beat_out(m_axis_tvalid && m_axis_tready),
        .stall(m_axis_tvalid && !m_axis_tready));
endmodule
