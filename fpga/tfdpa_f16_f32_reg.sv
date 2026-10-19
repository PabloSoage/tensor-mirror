// Registers around tfdpa_f16_f32 so that out-of-context synthesis reports the
// register-to-register delay of the combinational unit (one cycle, no pipelining yet).
module tfdpa_f16_f32_reg #(
    parameter int L = 4,
    parameter int F = 23,
    parameter int EZERO = -131
) (
    input  logic              clk,
    input  logic [L-1:0][15:0] a,
    input  logic [L-1:0][15:0] b,
    input  logic [31:0]        c,
    output logic [31:0]        d
);
    logic [L-1:0][15:0] a_q, b_q;
    logic [31:0] c_q, d_c;

    tfdpa_f16_f32 #(.L(L), .F(F), .EZERO(EZERO)) u (.a(a_q), .b(b_q), .c(c_q), .d(d_c));

    always_ff @(posedge clk) begin
        a_q <= a;
        b_q <= b;
        c_q <= c;
        d <= d_c;
    end
endmodule
