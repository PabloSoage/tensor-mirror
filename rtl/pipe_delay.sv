// D cycles of delay for a W-bit word while ce is 1 (a wire when D = 0).

module pipe_delay #(
    parameter int W = 1,
    parameter int D = 0
) (
    input  logic         clk,
    input  logic         ce,
    input  logic [W-1:0] x,
    output logic [W-1:0] y
);
    if (D == 0) begin : g_wire
        assign y = x;
    end else begin : g_reg
        logic [W-1:0] r [0:D-1];
        always_ff @(posedge clk)
            if (ce) begin
                r[0] <= x;
                for (int i = 1; i < D; i++) r[i] <= r[i-1];
            end
        assign y = r[D-1];
    end
endmodule
