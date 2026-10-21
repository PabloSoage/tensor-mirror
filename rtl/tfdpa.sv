// Truncated fused dot-product-add, every format a parameter: d = c + sum(a[k] * b[k]).
//
// MMA-Sim's T-FDPA (arXiv 2511.10909, Algorithm 7): exact products, every term aligned at
// the largest exponent and truncated toward zero to F fractional bits, an exact fixed-point
// sum, one rounding to the output. The golden model is python/tensormirror/tfdpa.py with
// the formats of python/tensormirror/formats.py; one set of parameters is one instruction of
// python/tensormirror/arch.py. rtl/tfdpa_f16.sv is the same datapath fixed to Volta's f16.
//
// A format is <x>E exponent bits, <x>M mantissa bits, <x>LSB its lowest bit in the word and
// <x>SPEC its special values: 0 IEEE-like (inf and NaN), 1 NaN only at all ones (OCP E4M3),
// 2 none (FP6, FP4). A and B words are AW and BW bits wide, c is 32 bits (an f16 c in
// c[15:0]), d is 32 bits with the result in its low DLSB + DE + DM + 1 bits. D is IEEE-like.
//
// PIPE = 0: combinational (clk unused). PIPE = 1: a register after each stage, one result
// per cycle while ce is 1, LATENCY cycles from a, b, c to d:
//   input -> decode -> multiply | max exponent -> align -> sum -> leading one -> shift -> round

module tfdpa #(
    parameter int L = 4,        // terms per group
    parameter int F = 23,       // fractional bits of the fused sum
    parameter int EZERO = -131, // exponent given to a zero term
    parameter int RNE = 0,      // output rounding: 0 toward zero, 1 to nearest even
    parameter int AW = 16, parameter int AE = 5, parameter int AM = 10,
    parameter int ALSB = 0, parameter int ASPEC = 0,
    parameter int BW = 16, parameter int BE = 5, parameter int BM = 10,
    parameter int BLSB = 0, parameter int BSPEC = 0,
    parameter int CE = 8, parameter int CM = 23, parameter int CLSB = 0,
    parameter int DE = 8, parameter int DM = 23, parameter int DLSB = 0,
    parameter int PIPE = 0      // 0 or 1
) (
    input  logic               clk,
    input  logic               ce,  // PIPE = 1: every stage holds while ce is 0
    input  logic [L-1:0][AW-1:0] a,
    input  logic [L-1:0][BW-1:0] b,
    input  logic [31:0]        c,
    output logic [31:0]        d
);
    /* verilator lint_off UNUSEDPARAM */
    localparam int LATENCY = PIPE != 0 ? 8 : 0;  // for instantiating code and the testbench
    /* verilator lint_on UNUSEDPARAM */
    localparam int ABIAS = (1 << (AE - 1)) - 1;
    localparam int BBIAS = (1 << (BE - 1)) - 1;
    localparam int CBIAS = (1 << (CE - 1)) - 1;
    localparam int DB = (1 << (DE - 1)) - 1;     // D's bias, also its largest exponent
    localparam int FP = AM + BM;                 // fractional bits of a product
    localparam int PW = AM + BM + 2;             // bits of a product of two significands
    localparam int W = F + 2;                    // aligned magnitude: every term is < 4
    localparam int N = L + 1;                    // terms: L products and c
    localparam int SW = W + $clog2(N) + 1;       // signed sum of N terms
    localparam int MW = $clog2(SW);              // bits of a leading-one position
    localparam int XW = 12;                      // signed exponents and shifts
    // a product or c's significand brought to F fractional bits: shifted left when it has
    // fewer, truncated when it has more (Ada's FP8 sums at F = 13)
    localparam int PX = W > PW ? W : PW;
    localparam int PSHL = F >= FP ? F - FP : 0, PSHR = F >= FP ? 0 : FP - F;
    localparam int CX = W > CM + 1 ? W : CM + 1;
    localparam int CSHL = F >= CM ? F - CM : 0, CSHR = F >= CM ? 0 : CM - F;
    localparam int DS = DLSB + DE + DM;          // D's sign bit
    localparam logic [31:0] NAN = 32'((64'd1 << DS) - 64'd1);
    localparam logic [31:0] PINF = 32'(((64'd1 << DE) - 64'd1) << (DM + DLSB));
    localparam logic [31:0] NINF = PINF | 32'(64'd1 << DS);

    typedef logic signed [XW-1:0] exp_t;
    typedef struct packed {                 // special values, carried down the pipeline
        logic nan, pos_inf, neg_inf;
    } spec_t;

    // A stage register: a flop when PIPE, a wire otherwise.
`define TFDPA_STAGE(q, dd) if (PIPE != 0) begin : g_``q always_ff @(posedge clk) if (ce) q <= dd; end \
                           else begin : g_``q assign q = dd; end

    // ---- 0. input ---------------------------------------------------------------------
    logic [L-1:0][AW-1:0] a0;
    logic [L-1:0][BW-1:0] b0;
    logic [31:0]          c0;
    `TFDPA_STAGE(a0, a)
    `TFDPA_STAGE(b0, b)
    `TFDPA_STAGE(c0, c)

    // ---- 1. decode ----------------------------------------------------------------------
    logic [L-1:0][AM:0] sa1_d, sa1;
    logic [L-1:0][BM:0] sb1_d, sb1;
    logic [W-1:0]       mc1_d, mc1;         // c's significand at F fractional bits
    exp_t [N-1:0]       ex1_d, ex1;
    logic [N-1:0]       neg1_d, neg1;
    spec_t              sp1_d, sp1;

    always_comb begin
        sp1_d = '0;
        for (int k = 0; k < L; k++) begin
            logic [AE-1:0] ea;
            logic [BE-1:0] eb;
            logic [AM-1:0] fa;
            logic [BM-1:0] fb;
            logic a_top, b_top, a_nan, b_nan, a_inf, b_inf, a_zero, b_zero;
            ea = a0[k][ALSB + AM +: AE];
            eb = b0[k][BLSB + BM +: BE];
            fa = a0[k][ALSB +: AM];
            fb = b0[k][BLSB +: BM];
            a_top = ea == '1;
            b_top = eb == '1;
            a_nan = (ASPEC == 0 && a_top && fa != '0) || (ASPEC == 1 && a_top && fa == '1);
            b_nan = (BSPEC == 0 && b_top && fb != '0) || (BSPEC == 1 && b_top && fb == '1);
            a_inf = ASPEC == 0 && a_top && fa == '0;
            b_inf = BSPEC == 0 && b_top && fb == '0;
            a_zero = ea == '0 && fa == '0;
            b_zero = eb == '0 && fb == '0;
            sa1_d[k] = {ea != '0, fa};
            sb1_d[k] = {eb != '0, fb};
            if (a_nan || b_nan || (a_inf && b_zero) || (b_inf && a_zero))
                sp1_d.nan = 1'b1;
            neg1_d[k] = a0[k][ALSB + AE + AM] ^ b0[k][BLSB + BE + BM];
            if ((a_inf || b_inf) && !(a_zero || b_zero) && !(a_nan || b_nan)) begin
                if (neg1_d[k]) sp1_d.neg_inf = 1'b1;
                else sp1_d.pos_inf = 1'b1;
            end
            // the product is zero exactly when one operand is (an inf operand's term is
            // never summed: the result is then inf or NaN)
            ex1_d[k] = (a_zero || b_zero) ? exp_t'(EZERO)
                     : exp_t'((ea == '0 ? 1 : int'(ea)) - ABIAS) + exp_t'((eb == '0 ? 1 : int'(eb)) - BBIAS);
        end
        begin
            logic [CE-1:0] ec;
            logic [CM-1:0] fc;
            logic [CM:0]   sc;
            ec = c0[CLSB + CM +: CE];
            fc = c0[CLSB +: CM];
            sc = {ec != '0, fc};
            if (ec == '1) begin
                if (fc != '0) sp1_d.nan = 1'b1;
                else if (c0[CLSB + CE + CM]) sp1_d.neg_inf = 1'b1;
                else sp1_d.pos_inf = 1'b1;
            end
            neg1_d[L] = c0[CLSB + CE + CM];
            // truncating c to F bits here and again when aligning drops the same bits as
            // one shift: both truncate a non-negative magnitude
            mc1_d = W'((CX'(sc) << CSHL) >> CSHR);
            ex1_d[L] = (sc == '0) ? exp_t'(EZERO) : exp_t'((ec == '0 ? 1 : int'(ec)) - CBIAS);
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
    logic [N-1:0][XW-1:0] sh2_d, sh2;       // emax - ex[k]
    exp_t                emax2_d, emax2;
    logic [N-1:0]        neg2;
    spec_t               sp2;

    always_comb begin
        for (int k = 0; k < L; k++) begin
            logic [PW-1:0] p;
            p = PW'(sa1[k]) * PW'(sb1[k]);              // exact
            mag2_d[k] = W'((PX'(p) << PSHL) >> PSHR);
        end
        mag2_d[L] = mc1;
        emax2_d = ex1[0];
        for (int k = 1; k < N; k++)
            if (ex1[k] > emax2_d) emax2_d = ex1[k];
        for (int k = 0; k < N; k++)
            sh2_d[k] = XW'(emax2_d - ex1[k]);
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
            t3_d[k] = (sh2[k] >= XW'(W)) ? '0
                    : W'(({1'b0, mag2[k]} + ((W + 1)'(1) << sh2[k] >> 1)) >> sh2[k]);
`else
            t3_d[k] = (sh2[k] >= XW'(W)) ? '0 : mag2[k] >> sh2[k];  // dropped bits: RZ
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
    logic [SW-2:0]        m5_d, m5;
    logic [MW-1:0]        msb5_d, msb5;
    logic                 sign5;
    exp_t                 emax5;
    logic signed [XW:0]   thr5_d, thr5, shsub5_d, shsub5;  // from emax alone, for stage 6
    spec_t                sp5;

    always_comb begin
        m5_d = sum4[SW-1] ? (SW-1)'(-sum4) : (SW-1)'(sum4);
        msb5_d = '0;
        for (int i = 0; i < SW - 1; i++)
            if (m5_d[i]) msb5_d = MW'(i);
        // stage 6's normal/subnormal test (msb + emax - F >= 1 - DB) and subnormal shift
        // depend on emax only, so they are ready before msb
        thr5_d = (XW + 1)'(1 - DB + F) - (XW + 1)'(emax4);
        shsub5_d = (XW + 1)'(1 - DB - DM + F) - (XW + 1)'(emax4);
    end
    `TFDPA_STAGE(m5, m5_d)
    `TFDPA_STAGE(msb5, msb5_d)
    `TFDPA_STAGE(sign5, sum4[SW-1])
    `TFDPA_STAGE(emax5, emax4)
    `TFDPA_STAGE(thr5, thr5_d)
    `TFDPA_STAGE(shsub5, shsub5_d)
    `TFDPA_STAGE(sp5, sp4)

    // ---- 6. shift to the output's last place -------------------------------------------------
    // The value is m5 * 2**(emax5 - F). It is shifted so that its last kept bit is the
    // output's last place: by sh = top - DM where top is the leading one's position
    // (normal), or to units of 2**(1 - DB - DM) (subnormal). The dropped part (rem) and its
    // half go to stage 7 with the kept part (q).
    logic signed [XW:0] e_res_d, e_res;     // exponent of the leading bit, before rounding
    int                 sh;                 // right shift to the output's last place
    logic [SW-1:0]      q_d, q6, rem_d, rem6, half_d, half6;
    logic               zero6;
    logic               sign6;
    spec_t              sp6;

    always_comb begin
        e_res_d = (XW + 1)'(msb5) + (XW + 1)'(emax5) - (XW + 1)'(F);
        if (int'(msb5) >= int'(thr5))                // e_res_d >= 1 - DB: normal
            sh = int'(msb5) - DM;
        else
            sh = int'(shsub5);                        // subnormal: units of 2**(1 - DB - DM)
        q_d = '0;
        rem_d = '0;
        half_d = '0;
        if (sh <= 0) begin
            q_d = (-sh < SW) ? SW'(m5) << (-sh) : '0;
        end else if (sh < SW) begin
            q_d = SW'(m5) >> sh;
            rem_d = SW'(m5) & ((SW'(1) << sh) - SW'(1));
            half_d = SW'(1) << (sh - 1);
        end else begin
            rem_d = SW'(m5);                          // everything is dropped
            half_d = '1;                              // more than any remainder: never rounds up
        end
    end
    `TFDPA_STAGE(q6, q_d)
    `TFDPA_STAGE(rem6, rem_d)
    `TFDPA_STAGE(half6, half_d)
    `TFDPA_STAGE(e_res, e_res_d)
    `TFDPA_STAGE(zero6, m5 == '0)
    `TFDPA_STAGE(sign6, sign5)
    `TFDPA_STAGE(sp6, sp5)

    // ---- 7. round to the output format, special values --------------------------------------
    // RZ drops rem; RNE adds one when it is above half, or exactly half with an odd q.
    logic signed [XW:0] e_out;              // exponent of the leading bit after rounding
    logic [XW:0]        e_field;            // biased exponent of a normal result
    logic [SW-1:0]      q;
    logic               up;
    logic [31:0]        fin, d7_d;

    always_comb begin
`ifdef MUT_RZ_OUT  // mutant: truncate a round-to-nearest output instead of rounding it
        up = RNE != 0 && 1'b0;
`else
        up = RNE != 0 && (rem6 > half6 || (rem6 == half6 && rem6 != '0 && q6[0]));
`endif
        q = q6 + SW'(up);
        e_out = e_res;
        if (e_res >= (XW + 1)'(1 - DB) && q[DM + 1]) begin  // rounding carried into a new binade
            q = q >> 1;
            e_out = e_res + (XW + 1)'(1);
        end
        e_field = (e_res >= (XW + 1)'(1 - DB) ? e_out : (XW + 1)'(1 - DB)) + (XW + 1)'(DB);
        if (zero6 || q == '0)
            fin = 32'd0;                             // +0, whatever the signs
        else if (e_out > (XW + 1)'(DB))
            fin = sign6 ? NINF : PINF;               // overflow gives infinity, also with RZ
        else if (q[DM])                              // normal (a subnormal can round up into one)
            fin = (32'(sign6) << DS) | (32'(e_field[DE-1:0]) << (DM + DLSB)) | (32'(q[DM-1:0]) << DLSB);
        else
            fin = (32'(sign6) << DS) | (32'(q[DM-1:0]) << DLSB);
        d7_d = sp6.nan || (sp6.pos_inf && sp6.neg_inf) ? NAN
             : sp6.pos_inf ? PINF
             : sp6.neg_inf ? NINF
             : fin;
    end
    `TFDPA_STAGE(d, d7_d)

`undef TFDPA_STAGE
endmodule
