`timescale 1ns/1ps

module naa_systolic_array #(
    parameter int N          = 16,
    parameter int DATA_WIDTH = 16,
    parameter int ACC_WIDTH  = 32
)(
    input  logic clk,
    input  logic rst_n,
    input  logic enable,

    input  logic signed [DATA_WIDTH-1:0] a_input [0:N-1],
    input  logic signed [DATA_WIDTH-1:0] b_input [0:N-1],

    input  logic a_valid [0:N-1],
    input  logic b_valid [0:N-1],

    output logic signed [ACC_WIDTH-1:0] c_output [0:N-1][0:N-1],
    output logic c_valid [0:N-1][0:N-1]
);

    // ------------------------------------------------------------
    // Systolic data paths
    // ------------------------------------------------------------

    wire signed [DATA_WIDTH-1:0]
        a_pipe [0:N-1][0:N];

    wire signed [DATA_WIDTH-1:0]
        b_pipe [0:N][0:N-1];

    wire
        a_valid_pipe [0:N-1][0:N];

    wire
        b_valid_pipe [0:N][0:N-1];

    // ------------------------------------------------------------
    // PE result wires
    //
    // Keep PE outputs on wires first, then explicitly assign them
    // to the module outputs. This avoids simulator-dependent
    // variable/net connection behavior.
    // ------------------------------------------------------------

    wire signed [ACC_WIDTH-1:0]
        acc_wire [0:N-1][0:N-1];

    wire
        acc_valid_wire [0:N-1][0:N-1];

    genvar r;
    genvar c;

    // ------------------------------------------------------------
    // A input boundary
    // ------------------------------------------------------------

    generate
        for (r = 0; r < N; r = r + 1) begin : A_BOUNDARY
            assign a_pipe[r][0]       = a_input[r];
            assign a_valid_pipe[r][0] = a_valid[r];
        end
    endgenerate

    // ------------------------------------------------------------
    // B input boundary
    // ------------------------------------------------------------

    generate
        for (c = 0; c < N; c = c + 1) begin : B_BOUNDARY
            assign b_pipe[0][c]       = b_input[c];
            assign b_valid_pipe[0][c] = b_valid[c];
        end
    endgenerate

    // ------------------------------------------------------------
    // PE grid
    // ------------------------------------------------------------

    generate
        for (r = 0; r < N; r = r + 1) begin : ROW

            for (c = 0; c < N; c = c + 1) begin : COL

                naa_systolic_pe #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACC_WIDTH(ACC_WIDTH)
                ) pe (
                    .clk(clk),
                    .rst_n(rst_n),
                    .enable(enable),

                    .a_in(a_pipe[r][c]),
                    .b_in(b_pipe[r][c]),

                    .a_valid_in(a_valid_pipe[r][c]),
                    .b_valid_in(b_valid_pipe[r][c]),

                    .a_out(a_pipe[r][c+1]),
                    .b_out(b_pipe[r+1][c]),

                    .a_valid_out(a_valid_pipe[r][c+1]),
                    .b_valid_out(b_valid_pipe[r+1][c]),

                    .acc_out(acc_wire[r][c]),
                    .acc_valid(acc_valid_wire[r][c])
                );

            end

        end
    endgenerate

    // ------------------------------------------------------------
    // Explicit result connections
    // ------------------------------------------------------------

    generate
        for (r = 0; r < N; r = r + 1) begin : RESULT_ROW
            for (c = 0; c < N; c = c + 1) begin : RESULT_COL

                assign c_output[r][c] = acc_wire[r][c];
                assign c_valid[r][c]  = acc_valid_wire[r][c];

            end
        end
    endgenerate

endmodule
