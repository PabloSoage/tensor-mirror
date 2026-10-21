// tm_regs - the AXI4-Lite registers and counters of the PL tops (rtl/tm_pl.v,
// rtl/tm_pl_x.v), 32 bits, 4 KB, one transaction at a time:
//   0x000 ID       "TMIR" (0x544D4952)      0x004 VERSION  the VERSION parameter
//   0x008 BUILD    the BUILD parameter (a commit, set by the build)
//   0x00C LATENCY  cycles from a case to its result while nothing stalls
//   0x010 F        fractional bits of the fused sum
//   0x014 CASES_IN beats accepted on s_axis     0x018 CASES_OUT beats sent on m_axis
//   0x01C STALLS   cycles m_axis held a result its consumer did not take
//   0x020 CONTROL  write 1 to bit 0 to clear the three counters
//   0x024 X0, 0x028 X1, 0x02C X2: words the top describes itself with (0 in tm_pl)

module tm_regs #(
    parameter [31:0] VERSION = 32'd1,
    parameter [31:0] BUILD = 32'h0,
    parameter [31:0] LATENCY = 32'd0,
    parameter [31:0] F = 32'd0,
    parameter [31:0] X0 = 32'd0,
    parameter [31:0] X1 = 32'd0,
    parameter [31:0] X2 = 32'd0
) (
    input  wire        clk,
    input  wire        pl_resetn,
    input  wire [11:0] s_axi_awaddr,
    input  wire        s_axi_awvalid,
    output reg         s_axi_awready,
    input  wire [31:0] s_axi_wdata,
    input  wire [3:0]  s_axi_wstrb,
    input  wire        s_axi_wvalid,
    output reg         s_axi_wready,
    output wire [1:0]  s_axi_bresp,
    output reg         s_axi_bvalid,
    input  wire        s_axi_bready,
    input  wire [11:0] s_axi_araddr,
    input  wire        s_axi_arvalid,
    output reg         s_axi_arready,
    output reg  [31:0] s_axi_rdata,
    output wire [1:0]  s_axi_rresp,
    output reg         s_axi_rvalid,
    input  wire        s_axi_rready,
    input  wire        beat_in,    // a beat accepted on s_axis
    input  wire        beat_out,   // a beat sent on m_axis
    input  wire        stall       // m_axis holds a result its consumer does not take
);
    // ---- counters --------------------------------------------------------------------
    reg [31:0] cases_in, cases_out, stalls;
    reg clear;
    always @(posedge clk) begin
        if (!pl_resetn || clear) begin
            cases_in <= 0;
            cases_out <= 0;
            stalls <= 0;
        end else begin
            if (beat_in) cases_in <= cases_in + 1;
            if (beat_out) cases_out <= cases_out + 1;
            if (stall) stalls <= stalls + 1;
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
                    10'h001: s_axi_rdata <= VERSION;
                    10'h002: s_axi_rdata <= BUILD;
                    10'h003: s_axi_rdata <= LATENCY;
                    10'h004: s_axi_rdata <= F;
                    10'h005: s_axi_rdata <= cases_in;
                    10'h006: s_axi_rdata <= cases_out;
                    10'h007: s_axi_rdata <= stalls;
                    10'h009: s_axi_rdata <= X0;
                    10'h00A: s_axi_rdata <= X1;
                    10'h00B: s_axi_rdata <= X2;
                    default: s_axi_rdata <= 32'd0;
                endcase
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 0;
            end
        end
    end
endmodule
