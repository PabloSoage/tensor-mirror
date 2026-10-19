// tm_pl - the PL top of the KV260 build: Volta's three m8n8k4 dot products, bit-exact,
// fed and drained by an AXI DMA.
//
//   s_axis  one case per beat, from the DMA's MM2S:
//             tdata[63:0]    a[0..3], f16, a[k] in bits 16k+15..16k
//             tdata[127:64]  b[0..3], f16
//             tdata[159:128] c: f32, or f16 in bits 143:128 for the f16-C units
//             tdata[255:160] unused
//   m_axis  one result per beat, to the DMA's S2MM, LATENCY beats after its case:
//             tdata[31:0]    d of f32.f16.f16.f32 (C f32, D f32, RZ)
//             tdata[63:32]   d of f32.f16.f16.f16 (C f16, D f32, RZ)
//             tdata[79:64]   d of f16.f16.f16.f16 (C f16, D f16, RNE)
//             tdata[127:80]  zero
//           tlast follows the case's tlast.
//   s_axi   AXI4-Lite registers (32 bits, 4 KB):
//             0x000 ID       "TMIR" (0x544D4952)      0x004 VERSION  1
//             0x008 BUILD    the BUILD parameter (a commit, set by the build)
//             0x00C LATENCY  cycles from a case to its result while nothing stalls
//             0x010 F        fractional bits of the fused sum (23)
//             0x014 CASES_IN beats accepted on s_axis     0x018 CASES_OUT beats sent on m_axis
//             0x01C STALLS   cycles m_axis held a result its consumer did not take
//             0x020 CONTROL  write 1 to bit 0 to clear the three counters
//
// The three units (rtl/tfdpa_f16.sv, PIPE = 1) share one clock enable: the pipeline
// moves when the output register is empty or being taken, and s_axis is ready exactly
// then, so a stall at the output holds every case in place and none is lost.

module tm_pl #(
    parameter [31:0] BUILD = 32'h0
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axi:s_axis:m_axis, ASSOCIATED_RESET pl_resetn" *)
    input  wire         clk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 pl_resetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire         pl_resetn,
    input  wire [11:0]  s_axi_awaddr,
    input  wire         s_axi_awvalid,
    output reg          s_axi_awready,
    input  wire [31:0]  s_axi_wdata,
    input  wire [3:0]   s_axi_wstrb,
    input  wire         s_axi_wvalid,
    output reg          s_axi_wready,
    output wire [1:0]   s_axi_bresp,
    output reg          s_axi_bvalid,
    input  wire         s_axi_bready,
    input  wire [11:0]  s_axi_araddr,
    input  wire         s_axi_arvalid,
    output reg          s_axi_arready,
    output reg  [31:0]  s_axi_rdata,
    output wire [1:0]   s_axi_rresp,
    output reg          s_axi_rvalid,
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
    localparam LATENCY = 7;  // tfdpa_f16 with PIPE = 1
    localparam F = 23;

    // ---- the datapath ----------------------------------------------------------------
    wire ce = !m_axis_tvalid || m_axis_tready;
`ifdef MUT_ALWAYS_READY  // mutant for the testbench: ignores back-pressure, so cases are lost
    assign s_axis_tready = 1'b1;
`else
    assign s_axis_tready = ce;
`endif
    wire [31:0] d_f32, d_f32c16, d_f16;

    tfdpa_f16 #(.L(4), .F(F), .C_F16(0), .D_F16(0), .PIPE(1)) u_f32 (
        .clk(clk), .ce(ce), .a(s_axis_tdata[63:0]), .b(s_axis_tdata[127:64]),
        .c(s_axis_tdata[159:128]), .d(d_f32));
    tfdpa_f16 #(.L(4), .F(F), .C_F16(1), .D_F16(0), .PIPE(1)) u_f32c16 (
        .clk(clk), .ce(ce), .a(s_axis_tdata[63:0]), .b(s_axis_tdata[127:64]),
        .c(s_axis_tdata[159:128]), .d(d_f32c16));
    tfdpa_f16 #(.L(4), .F(F), .C_F16(1), .D_F16(1), .PIPE(1)) u_f16 (
        .clk(clk), .ce(ce), .a(s_axis_tdata[63:0]), .b(s_axis_tdata[127:64]),
        .c(s_axis_tdata[159:128]), .d(d_f16));

    // valid and last travel beside the data, moving with the same enable
    reg [LATENCY-1:0] v, last;
    always @(posedge clk) begin
        if (!pl_resetn) begin
            v <= 0;
            last <= 0;
        end else if (ce) begin
            v <= {v[LATENCY-2:0], s_axis_tvalid};
            last <= {last[LATENCY-2:0], s_axis_tlast};
        end
    end
    assign m_axis_tvalid = v[LATENCY-1];
    assign m_axis_tlast = last[LATENCY-1];
    assign m_axis_tdata = {48'd0, d_f16[15:0], d_f32c16, d_f32};
    assign m_axis_tkeep = 16'hffff;

    // ---- counters --------------------------------------------------------------------
    reg [31:0] cases_in, cases_out, stalls;
    reg clear;
    always @(posedge clk) begin
        if (!pl_resetn || clear) begin
            cases_in <= 0;
            cases_out <= 0;
            stalls <= 0;
        end else begin
            if (s_axis_tvalid && s_axis_tready) cases_in <= cases_in + 1;
            if (m_axis_tvalid && m_axis_tready) cases_out <= cases_out + 1;
            if (m_axis_tvalid && !m_axis_tready) stalls <= stalls + 1;
        end
    end

    // ---- AXI4-Lite: one transaction at a time ----------------------------------------
    assign s_axi_bresp = 2'b00;
    assign s_axi_rresp = 2'b00;
    always @(posedge clk) begin
        if (!pl_resetn) begin
            s_axi_awready <= 0;
            s_axi_wready <= 0;
            s_axi_bvalid <= 0;
            s_axi_arready <= 0;
            s_axi_rvalid <= 0;
            s_axi_rdata <= 0;
            clear <= 0;
        end else begin
            clear <= 0;
            // write: take address and data together, answer OKAY
            s_axi_awready <= 0;
            s_axi_wready <= 0;
            if (s_axi_awvalid && s_axi_wvalid && !s_axi_awready && !s_axi_bvalid) begin
                s_axi_awready <= 1;
                s_axi_wready <= 1;
                s_axi_bvalid <= 1;
                if (s_axi_awaddr[11:2] == 10'h008 && s_axi_wstrb[0] && s_axi_wdata[0]) clear <= 1;
            end else if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 0;
            end
            // read
            s_axi_arready <= 0;
            if (s_axi_arvalid && !s_axi_arready && !s_axi_rvalid) begin
                s_axi_arready <= 1;
                s_axi_rvalid <= 1;
                case (s_axi_araddr[11:2])
                    10'h000: s_axi_rdata <= 32'h544D4952;
                    10'h001: s_axi_rdata <= 32'd1;
                    10'h002: s_axi_rdata <= BUILD;
                    10'h003: s_axi_rdata <= LATENCY;
                    10'h004: s_axi_rdata <= F;
                    10'h005: s_axi_rdata <= cases_in;
                    10'h006: s_axi_rdata <= cases_out;
                    10'h007: s_axi_rdata <= stalls;
                    default: s_axi_rdata <= 32'd0;
                endcase
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 0;
            end
        end
    end
endmodule
