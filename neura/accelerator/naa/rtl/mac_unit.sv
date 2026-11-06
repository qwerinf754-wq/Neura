`timescale 1ns/1ps

module naa_mac_unit #(
    parameter int DATA_WIDTH = 16,
    parameter int ACC_WIDTH  = 32
)(
    input  logic clk,
    input  logic rst_n,
    input  logic enable,

    input  logic signed [DATA_WIDTH-1:0] a,
    input  logic signed [DATA_WIDTH-1:0] b,

    input  logic signed [ACC_WIDTH-1:0] acc_in,
    output logic signed [ACC_WIDTH-1:0] acc_out
);

    logic signed [2*DATA_WIDTH-1:0] product;

    always_comb begin
        product = a * b;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_out <= '0;
        end else if (enable) begin
            acc_out <= acc_in + product;
        end
    end

endmodule
