// The largest of N signed values, as a binary tree of comparators with a register after
// every RE levels (PIPE = 1 and RE > 0); combinational otherwise. LAT = PIPE != 0 && RE > 0 ?
// $clog2(N) / RE : 0 cycles from x to y.

module pipe_max #(
    parameter int N = 5,
    parameter int XW = 12,
    parameter int RE = 2,
    parameter int PIPE = 0
) (
    input  logic                       clk,
    input  logic                       ce,
    input  logic signed [N-1:0][XW-1:0] x,
    output logic signed [XW-1:0]       y
);
    localparam int D = N > 1 ? $clog2(N) : 1;
    localparam int P = 1 << D;               // leaves, padded with copies of x[0]

    logic signed [XW-1:0] lv [0:D][0:P-1] /* verilator split_var */;  // one signal per node

    for (genvar i = 0; i < P; i++) begin : g_leaf
        assign lv[0][i] = x[i < N ? i : 0];
    end
    for (genvar l = 0; l < D; l++) begin : g_level
        localparam int M = P >> (l + 1);     // nodes at level l + 1
        for (genvar i = 0; i < P; i++) begin : g_node
            if (i < M) begin : g_used
                logic signed [XW-1:0] m;
                assign m = lv[l][2*i] >= lv[l][2*i+1] ? lv[l][2*i] : lv[l][2*i+1];
                if (PIPE != 0 && RE > 0 && (l + 1) % RE == 0) begin : g_reg
                    always_ff @(posedge clk) if (ce) lv[l+1][i] <= m;
                end else begin : g_wire
                    assign lv[l+1][i] = m;
                end
            end else begin : g_unused
                assign lv[l+1][i] = '0;
            end
        end
    end
    assign y = lv[D][0];
endmodule
