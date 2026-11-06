`timescale 1ns/1ps

module naa_systolic_pe #(
    parameter int DATA_WIDTH = 16,
    parameter int ACC_WIDTH  = 32
)(
    input  logic clk,
    input  logic rst_n,
    input  logic enable,

    input  logic signed [DATA_WIDTH-1:0] a_in,
    input  logic signed [DATA_WIDTH-1:0] b_in,

    input  logic a_valid_in,
    input  logic b_valid_in,

    output logic signed [DATA_WIDTH-1:0] a_out,
    output logic signed [DATA_WIDTH-1:0] b_out,

    output logic a_valid_out,
    output logic b_valid_out,

    output logic signed [ACC_WIDTH-1:0] acc_out,
    output logic acc_valid
);

    logic signed [2*DATA_WIDTH-1:0] product;

    assign product = a_in * b_in;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            a_out       <= '0;
            b_out       <= '0;
            a_valid_out <= 1'b0;
            b_valid_out <= 1'b0;
            acc_out     <= '0;
            acc_valid   <= 1'b0;
        end
        else if (enable) begin
            a_out       <= a_in;
            b_out       <= b_in;

            a_valid_out <= a_valid_in;
            b_valid_out <= b_valid_in;

            if (a_valid_in && b_valid_in) begin
                if (!acc_valid) begin
                    acc_out   <= product;
                    acc_valid <= 1'b1;
                end
                else begin
                    acc_out <= acc_out + product;
                end
            end
        end
        else begin
            a_valid_out <= 1'b0;
            b_valid_out <= 1'b0;
        end
    end

endmodule
