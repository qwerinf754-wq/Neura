`timescale 1ns/1ps

module bf16_mac_array #(
    parameter int N = 4
)(
    input  logic clk,
    input  logic rst_n,
    input  logic enable,

    input  logic [15:0] a_input [0:N-1],
    input  logic [15:0] b_input [0:N-1],

    input  logic a_valid [0:N-1],
    input  logic b_valid [0:N-1],

    output logic [31:0] c_output [0:N-1][0:N-1],
    output logic        c_valid  [0:N-1][0:N-1]
);

    /*
     * ============================================================
     * BF16 SYSTOLIC ARRAY
     *
     * A -> horizontal
     * B -> vertical
     *
     * ============================================================
     */

    wire [15:0] a_wire
        [0:N-1][0:N];

    wire [15:0] b_wire
        [0:N][0:N-1];

    wire a_valid_wire
        [0:N-1][0:N];

    wire b_valid_wire
        [0:N][0:N-1];

    wire [31:0] acc_wire
        [0:N-1][0:N-1];

    wire acc_valid_wire
        [0:N-1][0:N-1];


    genvar r;
    genvar c;

    /*
     * ------------------------------------------------------------
     * A input boundary
     * ------------------------------------------------------------
     */

    generate

        for (r = 0; r < N; r = r + 1) begin : GEN_A_INPUT

            assign a_wire[r][0]       = a_input[r];
            assign a_valid_wire[r][0] = a_valid[r];

        end

    endgenerate


    /*
     * ------------------------------------------------------------
     * B input boundary
     * ------------------------------------------------------------
     */

    generate

        for (c = 0; c < N; c = c + 1) begin : GEN_B_INPUT

            assign b_wire[0][c]       = b_input[c];
            assign b_valid_wire[0][c] = b_valid[c];

        end

    endgenerate


    /*
     * ------------------------------------------------------------
     * PE array
     * ------------------------------------------------------------
     */

    generate

        for (r = 0; r < N; r = r + 1) begin : GEN_ROWS

            for (c = 0; c < N; c = c + 1) begin : GEN_COLS

                bf16_pe pe (
                    .clk(clk),
                    .rst_n(rst_n),
                    .enable(enable),

                    .a_in(a_wire[r][c]),
                    .b_in(b_wire[r][c]),

                    .a_valid_in(a_valid_wire[r][c]),
                    .b_valid_in(b_valid_wire[r][c]),

                    .a_out(a_wire[r][c+1]),
                    .b_out(b_wire[r+1][c]),

                    .a_valid_out(a_valid_wire[r][c+1]),
                    .b_valid_out(b_valid_wire[r+1][c]),

                    .acc_out(acc_wire[r][c]),
                    .acc_valid(acc_valid_wire[r][c])
                );

                assign c_output[r][c] = acc_wire[r][c];
                assign c_valid[r][c]  = acc_valid_wire[r][c];

            end

        end

    endgenerate

endmodule
