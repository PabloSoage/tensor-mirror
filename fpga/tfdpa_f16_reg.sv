// Registers around tfdpa_f16 so that out-of-context synthesis reports its
// register-to-register delay: with PIPE = 0 that is the whole combinational unit, with
// PIPE = 1 the slowest pipeline stage.
module tfdpa_f16_reg #(
    parameter int L = 4,
    parameter int F = 23,
    parameter int C_F16 = 0,
    parameter int D_F16 = 0,
    parameter int PIPE = 0
) (
    input  logic               clk,
    input  logic [L-1:0][15:0] a,
    input  logic [L-1:0][15:0] b,
    input  logic [31:0]        c,
    output logic [31:0]        d
);
    logic [L-1:0][15:0] a_q, b_q;
    logic [31:0] c_q, d_c;

    tfdpa_f16 #(.L(L), .F(F), .C_F16(C_F16), .D_F16(D_F16), .PIPE(PIPE)) u (
        .clk(clk), .ce(1'b1), .a(a_q), .b(b_q), .c(c_q), .d(d_c));

    always_ff @(posedge clk) begin
        a_q <= a;
        b_q <= b;
        c_q <= c;
        d <= d_c;
    end
endmodule
