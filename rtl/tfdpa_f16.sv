// Truncated fused dot-product-add with f16 operands: d = c + sum(a[k] * b[k]).
//
// As Volta's HMMA computes it (MMA-Sim's T-FDPA, arXiv 2511.10909, Algorithm 7; Volta:
// L = 4, F = 23): exact products, every term aligned at the largest exponent and truncated
// toward zero to F fractional bits, an exact fixed-point sum, one rounding to the output.
// The golden model is python/tensormirror/tfdpa.py; this module follows it step by step.
//
// C_F16 = 0: c is f32 (in c[31:0]); 1: c is f16 (in c[15:0], c[31:16] ignored).
// D_F16 = 0: d is f32, rounded toward zero; 1: d is f16 (in d[15:0], d[31:16] = 0), rounded
// to nearest even. Volta's three mma.m8n8k4 variants are (C_F16, D_F16) = (0, 0), (1, 0)
// and (1, 1).
//
// PIPE = 0: combinational (clk unused). PIPE = 1: a register after each stage, one result
// per cycle, LATENCY cycles from a, b, c to d:
//   input -> decode -> multiply | max exponent -> align -> sum -> leading one -> output

module tfdpa_f16 #(
    parameter int L = 4,                             // terms per group
    parameter int F = 23,                            // fractional bits of the fused sum (>= 23)
    parameter int C_F16 = 0,
    parameter int D_F16 = 0,
    parameter int EZERO = C_F16 != 0 ? -20 : -131,   // exponent given to a zero term (Volta)
    parameter int PIPE = 0                           // 0 or 1
) (
    input  logic               clk,
    input  logic [L-1:0][15:0] a,
    input  logic [L-1:0][15:0] b,
    input  logic [31:0]        c,
    output logic [31:0]        d
);
    /* verilator lint_off UNUSEDPARAM */
    localparam int LATENCY = PIPE != 0 ? 7 : 0;  // for instantiating code and the testbench
    /* verilator lint_on UNUSEDPARAM */
    localparam int FP = 20;                 // fractional bits of an f16 x f16 product
    localparam int FC = C_F16 != 0 ? 10 : 23;  // fractional bits of c's significand
    localparam int W = F + 2;               // aligned magnitude: products are < 4
    localparam int SW = W + $clog2(L + 1) + 1;  // signed sum of L + 1 terms
    localparam int N = L + 1;               // terms: L products and c
    localparam int MW = $clog2(SW - 1);     // bits of a leading-one position
    // the output format: fraction bits, bias, largest exponent; NaN and infinity words
    localparam int DM = D_F16 != 0 ? 10 : 23;
    localparam int DB = D_F16 != 0 ? 15 : 127;
    localparam logic [31:0] NAN = D_F16 != 0 ? 32'h0000_7fff : 32'h7fff_ffff;
    localparam logic [31:0] PINF = D_F16 != 0 ? 32'h0000_7c00 : 32'h7f80_0000;
    localparam logic [31:0] NINF = D_F16 != 0 ? 32'h0000_fc00 : 32'hff80_0000;

    typedef logic signed [9:0] exp_t;       // covers EZERO .. 2 * 15 and the f32 range
    typedef struct packed {                 // special values, carried down the pipeline
        logic nan, pos_inf, neg_inf;
    } spec_t;

    // A stage register: a flop when PIPE, a wire otherwise.
`define TFDPA_STAGE(q, dd) if (PIPE != 0) begin : g_``q always_ff @(posedge clk) q <= dd; end \
                           else begin : g_``q assign q = dd; end

    // ---- 0. input ---------------------------------------------------------------------
    logic [L-1:0][15:0] a0, b0;
    logic [31:0]        c0;
    `TFDPA_STAGE(a0, a)
    `TFDPA_STAGE(b0, b)
    `TFDPA_STAGE(c0, c)

    // ---- 1. decode ----------------------------------------------------------------------
    logic [L-1:0][10:0] sa1_d, sb1_d, sa1, sb1;
    logic [W-1:0]       mc1_d, mc1;         // c's significand at F fractional bits
    exp_t [N-1:0]       ex1_d, ex1;
    logic [N-1:0]       neg1_d, neg1;
    spec_t              sp1_d, sp1;

    always_comb begin
        sp1_d = '0;
        for (int k = 0; k < L; k++) begin
            logic [4:0] ea, eb;
            logic a_inf, b_inf, a_zero, b_zero;
            ea = a0[k][14:10];
            eb = b0[k][14:10];
            sa1_d[k] = {ea != 5'd0, a0[k][9:0]};
            sb1_d[k] = {eb != 5'd0, b0[k][9:0]};
            a_inf = ea == 5'h1f && a0[k][9:0] == 10'd0;
            b_inf = eb == 5'h1f && b0[k][9:0] == 10'd0;
            a_zero = ea == 5'd0 && a0[k][9:0] == 10'd0;
            b_zero = eb == 5'd0 && b0[k][9:0] == 10'd0;
            if ((ea == 5'h1f && a0[k][9:0] != 10'd0) || (eb == 5'h1f && b0[k][9:0] != 10'd0)
                || (a_inf && b_zero) || (b_inf && a_zero))
                sp1_d.nan = 1'b1;
            neg1_d[k] = a0[k][15] ^ b0[k][15];
            if ((a_inf || b_inf) && !(a_zero || b_zero)) begin
                if (neg1_d[k]) sp1_d.neg_inf = 1'b1;
                else sp1_d.pos_inf = 1'b1;
            end
            // the product is zero exactly when one operand is (an inf operand's term is
            // never summed: the result is then inf or NaN)
            ex1_d[k] = (a_zero || b_zero) ? exp_t'(EZERO)
                     : exp_t'((ea == 5'd0 ? 1 : int'(ea)) - 15) + exp_t'((eb == 5'd0 ? 1 : int'(eb)) - 15);
        end
        if (C_F16 != 0) begin
            logic [4:0] ec;
            logic [10:0] sc;
            ec = c0[14:10];
            sc = {ec != 5'd0, c0[9:0]};
            if (ec == 5'h1f) begin
                if (c0[9:0] != 10'd0) sp1_d.nan = 1'b1;
                else if (c0[15]) sp1_d.neg_inf = 1'b1;
                else sp1_d.pos_inf = 1'b1;
            end
            neg1_d[L] = c0[15];
            mc1_d = W'(sc) << (F - FC);
            ex1_d[L] = (sc == 11'd0) ? exp_t'(EZERO) : exp_t'((ec == 5'd0 ? 1 : int'(ec)) - 15);
        end else begin
            logic [7:0] ec;
            logic [23:0] sc;
            ec = c0[30:23];
            sc = {ec != 8'd0, c0[22:0]};
            if (ec == 8'hff) begin
                if (c0[22:0] != 23'd0) sp1_d.nan = 1'b1;
                else if (c0[31]) sp1_d.neg_inf = 1'b1;
                else sp1_d.pos_inf = 1'b1;
            end
            neg1_d[L] = c0[31];
            mc1_d = W'(sc) << (F - FC);
            ex1_d[L] = (sc == 24'd0) ? exp_t'(EZERO) : exp_t'((ec == 8'd0 ? 1 : int'(ec)) - 127);
        end
    end
    `TFDPA_STAGE(sa1, sa1_d)
    `TFDPA_STAGE(sb1, sb1_d)
    `TFDPA_STAGE(mc1, mc1_d)
    `TFDPA_STAGE(ex1, ex1_d)
    `TFDPA_STAGE(neg1, neg1_d)
    `TFDPA_STAGE(sp1, sp1_d)

    // ---- 2. multiply; in parallel, the largest exponent and each term's shift ------------
    logic [N-1:0][W-1:0] mag2_d, mag2;      // significands at F fractional bits
    logic [N-1:0][9:0]   sh2_d, sh2;        // emax - ex[k], 0 .. 258
    exp_t                emax2_d, emax2;
    logic [N-1:0]        neg2;
    spec_t               sp2;

    always_comb begin
        for (int k = 0; k < L; k++)
            mag2_d[k] = W'(sa1[k] * sb1[k]) << (F - FP);  // exact; one DSP48E2 each
        mag2_d[L] = mc1;
        emax2_d = ex1[0];
        for (int k = 1; k < N; k++)
            if (ex1[k] > emax2_d) emax2_d = ex1[k];
        for (int k = 0; k < N; k++)
            sh2_d[k] = 10'(emax2_d - ex1[k]);
    end
    `TFDPA_STAGE(mag2, mag2_d)
    `TFDPA_STAGE(sh2, sh2_d)
    `TFDPA_STAGE(emax2, emax2_d)
    `TFDPA_STAGE(neg2, neg1)
    `TFDPA_STAGE(sp2, sp1)

    // ---- 3. align at the largest exponent, truncating toward zero -----------------------
    logic [N-1:0][W-1:0] t3_d, t3;
    exp_t                emax3;
    logic [N-1:0]        neg3;
    spec_t               sp3;

    always_comb begin
        for (int k = 0; k < N; k++) begin
`ifdef MUT_ROUND_ALIGN  // mutant for the testbench (NORMAS N8): round half up when aligning
            t3_d[k] = (sh2[k] >= 10'(W)) ? '0
                    : W'(({1'b0, mag2[k]} + ((W + 1)'(1) << sh2[k] >> 1)) >> sh2[k]);
`else
            t3_d[k] = (sh2[k] >= 10'(W)) ? '0 : mag2[k] >> sh2[k];  // dropped bits: RZ
`endif
        end
    end
    `TFDPA_STAGE(t3, t3_d)
    `TFDPA_STAGE(emax3, emax2)
    `TFDPA_STAGE(neg3, neg2)
    `TFDPA_STAGE(sp3, sp2)

    // ---- 4. exact fixed-point sum ---------------------------------------------------------
    logic signed [SW-1:0] sum4_d, sum4;
    exp_t                 emax4;
    spec_t                sp4;

    always_comb begin
        sum4_d = '0;
        for (int k = 0; k < N; k++)
            sum4_d = neg3[k] ? sum4_d - SW'(t3[k]) : sum4_d + SW'(t3[k]);
    end
    `TFDPA_STAGE(sum4, sum4_d)
    `TFDPA_STAGE(emax4, emax3)
    `TFDPA_STAGE(sp4, sp3)

    // ---- 5. magnitude and its leading one ---------------------------------------------------
    logic [SW-2:0]      m5_d, m5;
    logic [MW-1:0]      msb5_d, msb5;
    logic               sign5;
    exp_t               emax5;
    spec_t              sp5;

    always_comb begin
        m5_d = sum4[SW-1] ? (SW-1)'(-sum4) : (SW-1)'(sum4);
        msb5_d = '0;
        for (int i = 0; i < SW - 1; i++)
            if (m5_d[i]) msb5_d = MW'(i);
    end
    `TFDPA_STAGE(m5, m5_d)
    `TFDPA_STAGE(msb5, msb5_d)
    `TFDPA_STAGE(sign5, sum4[SW-1])
    `TFDPA_STAGE(emax5, emax4)
    `TFDPA_STAGE(sp5, sp4)

    // ---- 6. normalise and round to the output format, special values ------------------------
    // The value is m5 * 2**(emax5 - F). It is shifted so that its last kept bit is the
    // output's last place: by sh = top - DM where top is the leading one's position
    // (normal), or to units of 2**(1 - DB - DM) (subnormal). RZ drops the rest; RNE adds
    // one when the dropped part is above half, or exactly half with an odd kept part.
    logic signed [11:0] e_res, e_out;       // exponent of the leading bit, before/after rounding
    logic [11:0]        e_field;            // biased exponent of a normal result
    int                 sh;                 // right shift to the output's last place
    logic [SW-1:0]      q, rem, half;
    logic               up;
    logic [31:0]        fin, d6_d;

    always_comb begin
        e_res = 12'(msb5) + 12'(emax5) - 12'(F);
        if (e_res >= 12'(1 - DB))
            sh = int'(msb5) - DM;                     // normal
        else
            sh = (1 - DB - DM) - (int'(emax5) - F);   // subnormal: units of 2**(1 - DB - DM)
        q = '0;
        rem = '0;
        half = '0;
        if (sh <= 0) begin
            q = SW'(m5) << (-sh);
        end else if (sh < SW) begin
            q = SW'(m5) >> sh;
            rem = SW'(m5) & ((SW'(1) << sh) - SW'(1));
            half = SW'(1) << (sh - 1);
        end else begin
            rem = SW'(m5);                            // everything is dropped
            half = '1;                                // more than any remainder: never rounds up
        end
`ifdef MUT_RZ_OUT  // mutant: truncate the f16 output instead of rounding to nearest
        up = 1'b0;
`else
        up = D_F16 != 0 && (rem > half || (rem == half && rem != '0 && q[0]));
`endif
        q = q + SW'(up);
        e_out = e_res;
        if (e_res >= 12'(1 - DB) && q[DM + 1]) begin  // rounding carried into a new binade
            q = q >> 1;
            e_out = e_res + 12'sd1;
        end
        e_field = (e_res >= 12'(1 - DB) ? e_out : 12'(1 - DB)) + 12'(DB);
        if (m5 == '0 || q == '0)
            fin = 32'd0;                             // +0, whatever the signs
        else if (e_out > 12'(DB))
            fin = sign5 ? NINF : PINF;               // as MMA-Sim for RZ; not observable on Volta
        else if (q[DM])                              // normal (a subnormal can round up into one)
            fin = (32'(sign5) << (D_F16 != 0 ? 15 : 31))
                | (32'(e_field) << DM)
                | 32'(q[DM-1:0]);
        else
            fin = (32'(sign5) << (D_F16 != 0 ? 15 : 31)) | 32'(q[DM-1:0]);
        d6_d = sp5.nan || (sp5.pos_inf && sp5.neg_inf) ? NAN
             : sp5.pos_inf ? PINF
             : sp5.neg_inf ? NINF
             : fin;
    end
    `TFDPA_STAGE(d, d6_d)

`undef TFDPA_STAGE
endmodule
