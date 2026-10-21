// The sum of N signed values of SW bits (SW wide enough for the whole sum), as a binary
// adder tree with a register after every RE levels (PIPE = 1 and RE > 0); combinational
// otherwise. LAT = PIPE != 0 && RE > 0 ? $clog2(N) / RE : 0 cycles from x to y.

module pipe_sum #(
    parameter int N = 5,
    parameter int SW = 32,
    parameter int RE = 2,
    parameter int PIPE = 0
) (
    input  logic                        clk,
    input  logic                        ce,
    input  logic signed [N-1:0][SW-1:0] x,
    output logic signed [SW-1:0]        y
);
    localparam int D = N > 1 ? $clog2(N) : 1;
    localparam int P = 1 << D;               // leaves, padded with zeros

    logic signed [SW-1:0] lv [0:D][0:P-1] /* verilator split_var */;  // one signal per node

    for (genvar i = 0; i < P; i++) begin : g_leaf
        assign lv[0][i] = i < N ? x[i] : '0;
    end
    for (genvar l = 0; l < D; l++) begin : g_level
        localparam int M = P >> (l + 1);     // nodes at level l + 1
        for (genvar i = 0; i < P; i++) begin : g_node
            if (i < M) begin : g_used
                logic signed [SW-1:0] s;
                assign s = lv[l][2*i] + lv[l][2*i+1];
                if (PIPE != 0 && RE > 0 && (l + 1) % RE == 0) begin : g_reg
                    always_ff @(posedge clk) if (ce) lv[l+1][i] <= s;
                end else begin : g_wire
                    assign lv[l+1][i] = s;
                end
            end else begin : g_unused
                assign lv[l+1][i] = '0;
            end
        end
    end
    assign y = lv[D][0];
endmodule
