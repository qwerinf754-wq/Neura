`timescale 1ns/1ps

module test_bf16_pe;

    logic clk;
    logic rst_n;
    logic enable;

    logic [15:0] a_in;
    logic [15:0] b_in;
    logic a_valid_in;
    logic b_valid_in;

    logic [15:0] a_out;
    logic [15:0] b_out;
    logic a_valid_out;
    logic b_valid_out;

    logic [31:0] acc_out;
    logic acc_valid;

    bf16_pe dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),

        .a_in(a_in),
        .b_in(b_in),

        .a_valid_in(a_valid_in),
        .b_valid_in(b_valid_in),

        .a_out(a_out),
        .b_out(b_out),

        .a_valid_out(a_valid_out),
        .b_valid_out(b_valid_out),

        .acc_out(acc_out),
        .acc_valid(acc_valid)
    );

    always #5 clk = ~clk;


    task automatic check_value(
        input [31:0] expected,
        input [31:0] actual,
        input [255:0] name
    );
        begin

            if (actual !== expected) begin

                $display(
                    "[FAIL] %s expected=%h actual=%h",
                    name,
                    expected,
                    actual
                );

                $fatal;

            end
            else begin

                $display(
                    "[PASS] %s = %h",
                    name,
                    actual
                );

            end

        end
    endtask


    initial begin

        clk = 1'b0;

        rst_n = 1'b0;
        enable = 1'b0;

        a_in = 16'b0;
        b_in = 16'b0;

        a_valid_in = 1'b0;
        b_valid_in = 1'b0;


        $display("==============================================");
        $display("NAA BF16 PE TEST");
        $display("==============================================");


        repeat (3)
            @(posedge clk);


        rst_n = 1'b1;
        enable = 1'b1;


        /*
         * TEST 1
         *
         * 1.5 × 2.0 = 3.0
         *
         * BF16:
         * 1.5 = 3FC0
         * 2.0 = 4000
         * 3.0 = 4040
         *
         * FP32:
         * 3.0 = 40400000
         */

        @(negedge clk);

        a_in = 16'h3FC0;
        b_in = 16'h4000;

        a_valid_in = 1'b1;
        b_valid_in = 1'b1;

        @(posedge clk);
        #1;

        check_value(
            32'h40400000,
            acc_out,
            "1.5 x 2.0"
        );


        /*
         * TEST 2
         *
         * Previous:
         * 3.0
         *
         * New:
         * 2.0 × 2.0 = 4.0
         *
         * Expected:
         * 3.0 + 4.0 = 7.0
         *
         * FP32:
         * 7.0 = 40E00000
         */

        @(negedge clk);

        a_in = 16'h4000;
        b_in = 16'h4000;

        @(posedge clk);
        #1;

        check_value(
            32'h40E00000,
            acc_out,
            "3.0 + 4.0"
        );


        /*
         * Valid must remain asserted.
         */

        if (acc_valid !== 1'b1) begin

            $display(
                "[FAIL] acc_valid expected=1 actual=%b",
                acc_valid
            );

            $fatal;

        end

        $display("[PASS] accumulator valid");


        /*
         * Check systolic forwarding.
         */

        if (a_out !== 16'h4000) begin
            $display(
                "[FAIL] a_out expected=4000 actual=%h",
                a_out
            );
            $fatal;
        end

        if (b_out !== 16'h4000) begin
            $display(
                "[FAIL] b_out expected=4000 actual=%h",
                b_out
            );
            $fatal;
        end

        $display("[PASS] systolic forwarding");


        $display("----------------------------------------------");
        $display("[PASS] BF16 multiplication");
        $display("[PASS] FP32 accumulation");
        $display("[PASS] BF16 PE");
        $display("==============================================");

        $finish;

    end

endmodule
