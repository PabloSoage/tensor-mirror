// Truncated fused dot-product-add, f16 x f16 + f32 -> f32 with round toward zero.
//
// d = c + sum(a[k] * b[k]) as Volta's HMMA computes it (MMA-Sim's T-FDPA, arXiv
// 2511.10909, Algorithm 7; Volta: L = 4, F = 23): exact products, every term aligned at the
// largest exponent and truncated toward zero to F fractional bits, an exact fixed-point
// sum, one RZ rounding to f32. The golden model is python/tensormirror/tfdpa.py; this
// module follows it step by step.
//
// Combinational; pipelining comes with the board build.

module tfdpa_f16_f32 #(
    parameter int L = 4,         // terms per group
    parameter int F = 23,        // fractional bits of the fused sum (>= 23)
    parameter int EZERO = -131   // exponent given to a zero term
) (
    input  logic [L-1:0][15:0] a,
    input  logic [L-1:0][15:0] b,
    input  logic [31:0]        c,
    output logic [31:0]        d
);
    localparam int FP = 20;                 // fractional bits of an f16 x f16 product
    localparam int FC = 23;                 // fractional bits of an f32 significand
    localparam int W = F + 2;               // aligned magnitude: products are < 4
    localparam int SW = W + $clog2(L + 1) + 1;  // signed sum of L + 1 terms
    localparam int N = L + 1;               // terms: L products and c
    localparam logic [31:0] NAN = 32'h7fff_ffff;

    typedef logic signed [9:0] exp_t;       // covers EZERO .. 2 * 15 and the f32 range

    // ---- 1. decode and multiply -------------------------------------------------------
    logic [N-1:0][W-1:0] mag;               // significand placed at F - frac fractional bits
    exp_t [N-1:0]        ex;
    logic [N-1:0]        neg;
    logic                any_nan, pos_inf, neg_inf;

    always_comb begin
        any_nan = 1'b0;
        pos_inf = 1'b0;
        neg_inf = 1'b0;
        for (int k = 0; k < L; k++) begin
            logic [4:0] ea, eb;
            logic [10:0] sa, sb;
            logic a_inf, b_inf, a_zero, b_zero;
            logic [21:0] p;
            ea = a[k][14:10];
            eb = b[k][14:10];
            sa = {ea != 5'd0, a[k][9:0]};
            sb = {eb != 5'd0, b[k][9:0]};
            a_inf = ea == 5'h1f && a[k][9:0] == 10'd0;
            b_inf = eb == 5'h1f && b[k][9:0] == 10'd0;
            a_zero = ea == 5'd0 && a[k][9:0] == 10'd0;
            b_zero = eb == 5'd0 && b[k][9:0] == 10'd0;
            if ((ea == 5'h1f && a[k][9:0] != 10'd0) || (eb == 5'h1f && b[k][9:0] != 10'd0)
                || (a_inf && b_zero) || (b_inf && a_zero))
                any_nan = 1'b1;
            neg[k] = a[k][15] ^ b[k][15];
            if ((a_inf || b_inf) && !(a_zero || b_zero)) begin
                if (neg[k]) neg_inf = 1'b1;
                else pos_inf = 1'b1;
            end
            p = sa * sb;                     // exact; one DSP48E2 on the board
            mag[k] = W'(p) << (F - FP);
            ex[k] = (p == 22'd0) ? exp_t'(EZERO)
                  : exp_t'((ea == 5'd0 ? 1 : int'(ea)) - 15) + exp_t'((eb == 5'd0 ? 1 : int'(eb)) - 15);
        end
        begin
            logic [7:0] ec;
            logic [23:0] sc;
            ec = c[30:23];
            sc = {ec != 8'd0, c[22:0]};
            if (ec == 8'hff) begin
                if (c[22:0] != 23'd0) any_nan = 1'b1;
                else if (c[31]) neg_inf = 1'b1;
                else pos_inf = 1'b1;
            end
            neg[L] = c[31];
            mag[L] = W'(sc) << (F - FC);
            ex[L] = (sc == 24'd0) ? exp_t'(EZERO) : exp_t'((ec == 8'd0 ? 1 : int'(ec)) - 127);
        end
    end

    // ---- 2. align at the largest exponent, truncate, sum ----------------------------
    exp_t emax;
    logic signed [SW-1:0] sum;

    always_comb begin
        emax = ex[0];
        for (int k = 1; k < N; k++)
            if (ex[k] > emax) emax = ex[k];
        sum = '0;
        for (int k = 0; k < N; k++) begin
            logic [9:0] sh;                  // emax - ex[k], 0 .. 258
            logic [W-1:0] t;
            sh = 10'(emax - ex[k]);
`ifdef MUT_ROUND_ALIGN  // mutant for the testbench (NORMAS N8): round half up when aligning
            t = (sh >= 10'(W)) ? '0 : W'(({1'b0, mag[k]} + ((W + 1)'(1) << sh >> 1)) >> sh);
`else
            t = (sh >= 10'(W)) ? '0 : mag[k] >> sh;  // drops bits: truncation toward zero
`endif
            sum = neg[k] ? sum - SW'(t) : sum + SW'(t);
        end
    end

    // ---- 3. normalise and round toward zero to f32 ----------------------------------
    logic [SW-2:0] m, n, q;
    int unsigned   msb;
    int            s;
    logic signed [11:0] e_res;               // exponent of the result's leading bit
    logic [31:0]   fin;

    always_comb begin
        m = sum[SW-1] ? (SW-1)'(-sum) : (SW-1)'(sum);
        msb = 0;
        for (int i = 0; i < SW - 1; i++)
            if (m[i]) msb = i;
        e_res = 12'(msb) + 12'(emax) - 12'(F);
        n = (msb >= 23) ? m >> (msb - 23) : m << (23 - msb);  // normal: leading one at bit 23
        s = int'(emax) - F + 149;                            // subnormal: units of 2**-149
        q = (s >= 0) ? m << s : m >> (-s);
        if (m == '0)
            fin = 32'd0;                     // +0, whatever the signs
        else if (e_res > 12'sd127)
            fin = {sum[SW-1], 8'hff, 23'd0}; // as MMA-Sim; not observable on Volta (H-002)
        else if (e_res >= -12'sd126)
            fin = {sum[SW-1], 8'(e_res + 12'sd127), n[22:0]};
        else
            fin = {sum[SW-1], 8'd0, q[22:0]};
    end

    assign d = any_nan || (pos_inf && neg_inf) ? NAN
             : pos_inf ? 32'h7f80_0000
             : neg_inf ? 32'hff80_0000
             : fin;
endmodule
