// Group-scaled fused dot-product-add: RTX Blackwell's mma.block_scale with kind::mxf4 and
// kind::mxf4nvf4 (packed E2M1, K = 64), as MMA-Sim's GST-FDPA (python/tensormirror/
// blockscale.py gst_fdpa, which matches the RTX 5070 Ti bit for bit, H-010):
//
// 1. the products of each group of G elements are summed exactly, as integers;
// 2. each group's sum is multiplied by its scale pair's significand and aligned by the scale
//    pair's exponent alone; a group whose products are all zero, or whose scale is zero,
//    takes EZERO (a sum that cancels keeps the scale's exponent), and so does a zero c;
// 3. the K / G group terms and c are aligned at the largest exponent, truncated toward zero
//    to F fractional bits, summed exactly and rounded toward zero to f32.
//
// Scales: NB per row of A and column of B, one per K / NB elements. UE8M0 (UE4M3 = 0) is
// 2**(x - 127), 0xFF NaN; UE4M3 (UE4M3 = 1) is E4M3 with the sign bit ignored, 0x7F NaN.
// Both are taken as a 4-bit significand with 3 fractional bits (UE8M0 is 8 = 1.000), which
// truncates exactly as the model's (1, 0, e).
//
// a[k], b[k] are E2M1 nibbles (sign, 2 exponent bits, 1 mantissa bit; no inf or NaN). Stages
// 3 to 7 are rtl/tfdpa.sv's with D fixed to f32 RZ. PIPE as in rtl/tfdpa.sv (8 stages).
// XP = 1 makes each group's sum and the sum of the terms pipelined adder trees
// (rtl/pipe_sum.sv, a register every 2 levels), and SP = 1 puts a register in stage 6,
// between the shift amount and the shift, as in rtl/tfdpa.sv: LATENCY grows from 8 to
// 8 + $clog2(G) / 2 + $clog2(K / G + 1) / 2 + 1 (12 for K = 64 and G = 16 or 8).

module gst_fdpa #(
    parameter int K = 64,       // elements
    parameter int G = 16,       // elements per exactly summed group
    parameter int NB = 2,       // scale factors per row of A / column of B
    parameter int UE4M3 = 0,    // scale format: 0 UE8M0, 1 UE4M3
    parameter int F = 35,       // fractional bits of the fused sum
    parameter int EZERO = -139, // exponent given to a zero term
    parameter int XP = 0,       // 1: the group sums and the sum of the terms as pipelined trees
    parameter int SP = 0,       // 1: stage 6 split in two
    parameter int PIPE = 0      // 0 or 1
) (
    input  logic                clk,
    input  logic                ce,  // PIPE = 1: every stage holds while ce is 0
    input  logic [K-1:0][3:0]   a,
    input  logic [K-1:0][3:0]   b,
    input  logic [31:0]         c,
    input  logic [NB-1:0][7:0]  sfa,
    input  logic [NB-1:0][7:0]  sfb,
    output logic [31:0]         d
);
    localparam int NG = K / G;                   // group terms
    localparam int N = NG + 1;                   // terms: the groups and c
    // with XP, cycles in the group adder trees and in the terms' adder tree; with SP, the
    // cycle added in stage 6
    localparam int LG = PIPE != 0 && XP != 0 ? $clog2(G) / 2 : 0;
    localparam int LS = PIPE != 0 && XP != 0 ? $clog2(N) / 2 : 0;
    localparam int LN = PIPE != 0 && SP != 0 ? 1 : 0;
    /* verilator lint_off UNUSEDPARAM */
    localparam int LATENCY = PIPE != 0 ? 8 + LG + LS + LN : 0;  // for instantiating code and the testbench
    /* verilator lint_on UNUSEDPARAM */
    localparam int BLOCK = K / NB;               // elements per scale factor
    localparam int PB = 8;                       // a product |x * y| <= 144 in units of 2**-2
    localparam int GSW = PB + $clog2(G) + 1;     // signed sum of a group's products
    localparam int TSW = GSW + 8;                // times the scale pair's significand (<= 225)
    localparam int TF = 2 + 6;                   // fractional bits of a group term
    localparam int W = F + TSW - 1 - TF + 1;     // aligned magnitude of any term (c is < 2)
    localparam int SW = W + $clog2(N) + 1;       // signed sum of N terms
    localparam int MW = $clog2(SW);              // bits of a leading-one position
    localparam int XW = 12;                      // signed exponents and shifts
    localparam int DM = 23, DB = 127;            // d is f32
    localparam logic [31:0] NAN = 32'h7fff_ffff;
    localparam logic [31:0] PINF = 32'h7f80_0000;
    localparam logic [31:0] NINF = 32'hff80_0000;

    typedef logic signed [XW-1:0] exp_t;
    typedef struct packed {                 // special values, carried down the pipeline
        logic nan, pos_inf, neg_inf;
    } spec_t;

`define GST_STAGE(q, dd) if (PIPE != 0) begin : g_``q always_ff @(posedge clk) if (ce) q <= dd; end \
                         else begin : g_``q assign q = dd; end

    // ---- 0. input ---------------------------------------------------------------------
    logic [K-1:0][3:0]  a0, b0;
    logic [31:0]        c0;
    logic [NB-1:0][7:0] sfa0, sfb0;
    `GST_STAGE(a0, a)
    `GST_STAGE(b0, b)
    `GST_STAGE(c0, c)
    `GST_STAGE(sfa0, sfa)
    `GST_STAGE(sfb0, sfb)

    // ---- 1. decode: the exact products, the scale pairs and c -----------------------------
    logic [K-1:0][PB:0] p1_d, p1;           // signed x * y in units of 2**-2 (packed: read with $signed)
    logic [NG-1:0]             nz1_d, nz1;  // a group has a non-zero product
    logic [NB-1:0][7:0]        ss1_d, ss1;  // scale pair significand, 6 fractional bits
    exp_t [NB-1:0]             se1_d, se1;  // scale pair exponent
    logic [23:0]               sc1_d, sc1;
    exp_t                      ec1_d, ec1;
    logic                      negc1_d, negc1;
    spec_t                     sp1_d, sp1;

    function automatic logic [3:0] e2m1_sig(input logic [3:0] x);
        return {2'b0, x[2:1] != 2'd0, x[0]};
    endfunction
    function automatic int e2m1_exp(input logic [3:0] x);  // emin = 0 (bias 1)
        return x[2:1] == 2'd0 ? 0 : int'(x[2:1]) - 1;
    endfunction

    always_comb begin
        sp1_d = '0;
        for (int k = 0; k < K; k++) begin
            logic [7:0] m;
            m = 8'(e2m1_sig(a0[k]) * e2m1_sig(b0[k])) << (e2m1_exp(a0[k]) + e2m1_exp(b0[k]));
            p1_d[k] = (a0[k][3] ^ b0[k][3]) ? -(PB + 1)'(m) : (PB + 1)'(m);
        end
        for (int g = 0; g < NG; g++) begin
            nz1_d[g] = 1'b0;
            for (int k = g * G; k < g * G + G; k++)
                nz1_d[g] = nz1_d[g] | (p1_d[k] != '0);
        end
        for (int j = 0; j < NB; j++) begin
            logic [3:0] sa, sb;
            int ea, eb;
            logic nan_a, nan_b;
            if (UE4M3 != 0) begin  // E4M3 without its sign: (e, m), emin -6, 0x7F NaN
                nan_a = sfa0[j][6:0] == 7'h7f;
                nan_b = sfb0[j][6:0] == 7'h7f;
                sa = {sfa0[j][6:3] != 4'd0, sfa0[j][2:0]};
                sb = {sfb0[j][6:3] != 4'd0, sfb0[j][2:0]};
                ea = sfa0[j][6:3] == 4'd0 ? -6 : int'(sfa0[j][6:3]) - 7;
                eb = sfb0[j][6:3] == 4'd0 ? -6 : int'(sfb0[j][6:3]) - 7;
            end else begin         // 2**(x - 127) as 1.000, 0xFF NaN
                nan_a = sfa0[j] == 8'hff;
                nan_b = sfb0[j] == 8'hff;
                sa = 4'd8;
                sb = 4'd8;
                ea = int'(sfa0[j]) - 127;
                eb = int'(sfb0[j]) - 127;
            end
            if (nan_a || nan_b) sp1_d.nan = 1'b1;
            ss1_d[j] = 8'(sa) * 8'(sb);
            se1_d[j] = exp_t'(ea + eb);
        end
        sc1_d = {c0[30:23] != 8'd0, c0[22:0]};
        ec1_d = sc1_d == '0 ? exp_t'(EZERO) : exp_t'((c0[30:23] == 8'd0 ? 1 : int'(c0[30:23])) - 127);
        negc1_d = c0[31];
        if (c0[30:23] == 8'hff) begin
            if (c0[22:0] != 23'd0) sp1_d.nan = 1'b1;
            else if (c0[31]) sp1_d.neg_inf = 1'b1;
            else sp1_d.pos_inf = 1'b1;
        end
    end
    `GST_STAGE(p1, p1_d)
    `GST_STAGE(nz1, nz1_d)
    `GST_STAGE(ss1, ss1_d)
    `GST_STAGE(se1, se1_d)
    `GST_STAGE(sc1, sc1_d)
    `GST_STAGE(ec1, ec1_d)
    `GST_STAGE(negc1, negc1_d)
    `GST_STAGE(sp1, sp1_d)

    // ---- 2. group sums times their scales; the largest exponent and each term's shift ------
    logic [N-1:0][W-1:0]  mag2_d, mag2;     // terms at F fractional bits
    logic [N-1:0]         neg2_d, neg2;
    logic [N-1:0][XW-1:0] sh2_d, sh2;
    exp_t                 emax2_d, emax2;
    spec_t                sp2;

    logic signed [NG-1:0][GSW-1:0] total;  // each group's exact sum
    logic [NG-1:0]             nzx;
    logic [NB-1:0][7:0]        ssx;
    exp_t [NB-1:0]             sex;
    logic [23:0]               scx;
    exp_t                      ecx;
    logic                      negcx;
    spec_t                     spx;

    for (genvar g = 0; g < NG; g++) begin : g_group
        logic signed [G-1:0][GSW-1:0] x;
        always_comb
            for (int k = 0; k < G; k++)
                x[k] = GSW'($signed(p1[g * G + k]));
        if (XP != 0) begin : g_tree  // an adder tree with registers inside (LG cycles)
            pipe_sum #(.N(G), .SW(GSW), .RE(2), .PIPE(PIPE)) u_sum (
                .clk(clk), .ce(ce), .x(x), .y(total[g]));
        end else begin : g_chain
            always_comb begin
                total[g] = '0;
                for (int k = 0; k < G; k++)
                    total[g] = total[g] + x[k];
            end
        end
    end
    pipe_delay #(.W(NG), .D(LG)) u_dnz (.clk(clk), .ce(ce), .x(nz1), .y(nzx));
    pipe_delay #(.W(8 * NB), .D(LG)) u_dss (.clk(clk), .ce(ce), .x(ss1), .y(ssx));
    pipe_delay #(.W(XW * NB), .D(LG)) u_dse (.clk(clk), .ce(ce), .x(se1), .y(sex));
    pipe_delay #(.W(24), .D(LG)) u_dsc (.clk(clk), .ce(ce), .x(sc1), .y(scx));
    pipe_delay #(.W(XW), .D(LG)) u_dec (.clk(clk), .ce(ce), .x(ec1), .y(ecx));
    pipe_delay #(.W(1), .D(LG)) u_dnegc (.clk(clk), .ce(ce), .x(negc1), .y(negcx));
    pipe_delay #(.W(3), .D(LG)) u_dsp1 (.clk(clk), .ce(ce), .x(sp1), .y(spx));

    always_comb begin
        exp_t ex[N];
        for (int g = 0; g < NG; g++) begin
            logic signed [TSW-1:0] t;
            logic [TSW-1:0]        tm;
            int                    j;
            j = g * G / BLOCK;
            t = TSW'($signed(total[g])) * $signed({1'b0, ssx[j]});  // an element of a packed array is unsigned
            neg2_d[g] = t < 0;
            tm = t < 0 ? TSW'(-t) : TSW'(t);
            mag2_d[g] = W'(tm) << (F - TF);
`ifdef MUT_GST_ZERO_RULE  // mutant: a sum that cancels to zero also takes EZERO
            ex[g] = (total[g] == '0 || ssx[j] == '0) ? exp_t'(EZERO) : sex[j];
`else
            ex[g] = (!nzx[g] || ssx[j] == '0) ? exp_t'(EZERO) : sex[j];
`endif
        end
        neg2_d[NG] = negcx;
        mag2_d[NG] = W'(scx) << (F - DM);
        ex[NG] = ecx;
        emax2_d = ex[0];
        for (int k = 1; k < N; k++)
            if (ex[k] > emax2_d) emax2_d = ex[k];
        for (int k = 0; k < N; k++)
            sh2_d[k] = XW'(emax2_d - ex[k]);
    end
    `GST_STAGE(mag2, mag2_d)
    `GST_STAGE(neg2, neg2_d)
    `GST_STAGE(sh2, sh2_d)
    `GST_STAGE(emax2, emax2_d)
    `GST_STAGE(sp2, spx)

    // ---- 3. align at the largest exponent, truncating toward zero -----------------------
    logic [N-1:0][W-1:0] t3_d, t3;
    exp_t                emax3;
    logic [N-1:0]        neg3;
    spec_t               sp3;

    always_comb
        for (int k = 0; k < N; k++)
            t3_d[k] = (sh2[k] >= XW'(W)) ? '0 : mag2[k] >> sh2[k];  // dropped bits: RZ
    `GST_STAGE(t3, t3_d)
    `GST_STAGE(emax3, emax2)
    `GST_STAGE(neg3, neg2)
    `GST_STAGE(sp3, sp2)

    // ---- 4. exact fixed-point sum ---------------------------------------------------------
    logic signed [SW-1:0] sum4_d, sum4;
    exp_t                 emax4;
    spec_t                sp4;

    exp_t                 emaxs;
    spec_t                sps;

    if (XP != 0) begin : g_sum_tree  // an adder tree with registers inside (LS cycles)
        logic signed [N-1:0][SW-1:0] terms;
        always_comb
            for (int k = 0; k < N; k++)
                terms[k] = neg3[k] ? -SW'(t3[k]) : SW'(t3[k]);
        pipe_sum #(.N(N), .SW(SW), .RE(2), .PIPE(PIPE)) u_sum (
            .clk(clk), .ce(ce), .x(terms), .y(sum4_d));
    end else begin : g_sum_chain
        always_comb begin
            sum4_d = '0;
            for (int k = 0; k < N; k++)
                sum4_d = neg3[k] ? sum4_d - SW'(t3[k]) : sum4_d + SW'(t3[k]);
        end
    end
    pipe_delay #(.W(XW), .D(LS)) u_demax (.clk(clk), .ce(ce), .x(emax3), .y(emaxs));
    pipe_delay #(.W(3), .D(LS)) u_dsp4 (.clk(clk), .ce(ce), .x(sp3), .y(sps));
    `GST_STAGE(sum4, sum4_d)
    `GST_STAGE(emax4, emaxs)
    `GST_STAGE(sp4, sps)

    // ---- 5. magnitude and its leading one ---------------------------------------------------
    logic [SW-2:0]      m5_d, m5;
    logic [MW-1:0]      msb5_d, msb5;
    logic               sign5;
    exp_t               emax5;
    logic signed [XW:0] thr5_d, thr5, shsub5_d, shsub5;
    spec_t              sp5;

    always_comb begin
        m5_d = sum4[SW-1] ? (SW-1)'(-sum4) : (SW-1)'(sum4);
        msb5_d = '0;
        for (int i = 0; i < SW - 1; i++)
            if (m5_d[i]) msb5_d = MW'(i);
        thr5_d = (XW + 1)'(1 - DB + F) - (XW + 1)'(emax4);
        shsub5_d = (XW + 1)'(1 - DB - DM + F) - (XW + 1)'(emax4);
    end
    `GST_STAGE(m5, m5_d)
    `GST_STAGE(msb5, msb5_d)
    `GST_STAGE(sign5, sum4[SW-1])
    `GST_STAGE(emax5, emax4)
    `GST_STAGE(thr5, thr5_d)
    `GST_STAGE(shsub5, shsub5_d)
    `GST_STAGE(sp5, sp4)

    // ---- 6. shift to the output's last place -------------------------------------------------
    // With SP, the shift amount is registered before the shift.
    logic signed [XW:0] e_res_d, e_resx, e_res;
    logic signed [XW:0] sh_d, shx;
    int                 sh;
    logic [SW-2:0]      m5x;
    logic               sign5x;
    spec_t              sp5x;
    logic [SW-1:0]      q_d, q6;
    logic               zero6;
    logic               sign6;
    spec_t              sp6;

    always_comb begin
        e_res_d = (XW + 1)'(msb5) + (XW + 1)'(emax5) - (XW + 1)'(F);
        if (int'(msb5) >= int'(thr5))
            sh_d = (XW + 1)'(msb5) - (XW + 1)'(DM);
        else
            sh_d = shsub5;
    end
    pipe_delay #(.W(XW + 1), .D(LN)) u_dsh (.clk(clk), .ce(ce), .x(sh_d), .y(shx));
    pipe_delay #(.W(XW + 1), .D(LN)) u_dres (.clk(clk), .ce(ce), .x(e_res_d), .y(e_resx));
    pipe_delay #(.W(SW - 1), .D(LN)) u_dm5 (.clk(clk), .ce(ce), .x(m5), .y(m5x));
    pipe_delay #(.W(1), .D(LN)) u_dsign5 (.clk(clk), .ce(ce), .x(sign5), .y(sign5x));
    pipe_delay #(.W(3), .D(LN)) u_dsp5 (.clk(clk), .ce(ce), .x(sp5), .y(sp5x));

    always_comb begin
        sh = int'(shx);
        if (sh <= 0)
            q_d = (-sh < SW) ? SW'(m5x) << (-sh) : '0;
        else
            q_d = (sh < SW) ? SW'(m5x) >> sh : '0;  // RZ: the dropped bits do not matter
    end
    `GST_STAGE(q6, q_d)
    `GST_STAGE(e_res, e_resx)
    `GST_STAGE(zero6, m5x == '0)
    `GST_STAGE(sign6, sign5x)
    `GST_STAGE(sp6, sp5x)

    // ---- 7. pack the f32 result, special values ----------------------------------------------
    logic [XW:0] e_field;
    logic [31:0] fin, d7_d;

    always_comb begin
        e_field = (e_res >= (XW + 1)'(1 - DB) ? e_res : (XW + 1)'(1 - DB)) + (XW + 1)'(DB);
        if (zero6 || q6 == '0)
            fin = 32'd0;                             // +0, whatever the signs
        else if (e_res > (XW + 1)'(DB))
            fin = sign6 ? NINF : PINF;               // overflow gives infinity (H-010)
        else if (q6[DM])
            fin = {sign6, e_field[7:0], q6[DM-1:0]};
        else
            fin = {sign6, 8'd0, q6[DM-1:0]};
        d7_d = sp6.nan || (sp6.pos_inf && sp6.neg_inf) ? NAN
             : sp6.pos_inf ? PINF
             : sp6.neg_inf ? NINF
             : fin;
    end
    `GST_STAGE(d, d7_d)

`undef GST_STAGE
endmodule
