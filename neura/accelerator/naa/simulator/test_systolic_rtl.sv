`timescale 1ns/1ps

module test_systolic_rtl;

    parameter int N          = 16;
    parameter int DATA_WIDTH = 16;
    parameter int ACC_WIDTH  = 32;

    logic clk;
    logic rst_n;
    logic enable;

    logic signed [DATA_WIDTH-1:0] a_input [0:N-1];
    logic signed [DATA_WIDTH-1:0] b_input [0:N-1];

    logic a_valid [0:N-1];
    logic b_valid [0:N-1];

    wire signed [ACC_WIDTH-1:0] c_output [0:N-1][0:N-1];
    wire c_valid [0:N-1][0:N-1];

    integer A [0:N-1][0:N-1];
    integer B [0:N-1][0:N-1];
    integer expected [0:N-1][0:N-1];

    integer cycle;
    integer i;
    integer j;
    integer k;
    integer x;
    integer kk;

    naa_systolic_array #(
        .N(N),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),

        .a_input(a_input),
        .b_input(b_input),

        .a_valid(a_valid),
        .b_valid(b_valid),

        .c_output(c_output),
        .c_valid(c_valid)
    );

    always #5 clk = ~clk;

    task automatic clear_inputs;
        begin
            for (x = 0; x < N; x = x + 1) begin
                a_input[x] = '0;
                b_input[x] = '0;
                a_valid[x] = 1'b0;
                b_valid[x] = 1'b0;
            end
        end
    endtask

    task automatic build_matrices;
        begin
            for (x = 0; x < N; x = x + 1) begin
                for (j = 0; j < N; j = j + 1) begin
                    A[x][j] = (x + 1) * 2 + j;
                    B[x][j] = (x + 1) * 3 - j;
                end
            end

            for (x = 0; x < N; x = x + 1) begin
                for (j = 0; j < N; j = j + 1) begin
                    expected[x][j] = 0;

                    for (k = 0; k < N; k = k + 1) begin
                        expected[x][j] =
                            expected[x][j] +
                            A[x][k] * B[k][j];
                    end
                end
            end
        end
    endtask

    task automatic drive_wavefront;
        begin
            clear_inputs();

            // A: left -> right
            for (x = 0; x < N; x = x + 1) begin
                kk = cycle - x;

                if ((kk >= 0) && (kk < N)) begin
                    a_input[x] = A[x][kk];
                    a_valid[x] = 1'b1;
                end
            end

            // B: top -> bottom
            for (x = 0; x < N; x = x + 1) begin
                kk = cycle - x;

                if ((kk >= 0) && (kk < N)) begin
                    b_input[x] = B[kk][x];
                    b_valid[x] = 1'b1;
                end
            end
        end
    endtask

    initial begin

        clk    = 1'b0;
        rst_n  = 1'b0;
        enable = 1'b0;

        clear_inputs();
        build_matrices();

        $display("");
        $display("==============================================");
        $display("NAA RTL SYSTOLIC TEST");
        $display("N = %0d", N);
        $display("==============================================");

        // ------------------------------------------------
        // RESET
        // ------------------------------------------------

        $display("RESET: asserting rst_n=0");

        rst_n  = 1'b0;
        enable = 1'b0;

        repeat (4) begin
            @(posedge clk);
            #1;

            $display(
                "reset: c00=%0d valid=%0b",
                $signed(c_output[0][0]),
                c_valid[0][0]
            );
        end

        // ------------------------------------------------
        // RELEASE RESET
        // ------------------------------------------------

        @(negedge clk);

        rst_n  = 1'b1;
        enable = 1'b1;

        $display("RESET: released");
        $display("");

        // ------------------------------------------------
        // FEED WAVEFRONT
        // ------------------------------------------------

        for (cycle = 0; cycle < (3*N + 4); cycle = cycle + 1) begin

            @(negedge clk);

            drive_wavefront();

            $display(
                "cycle=%0d A0=%0d vA0=%0b B0=%0d vB0=%0b",
                cycle,
                a_input[0],
                a_valid[0],
                b_input[0],
                b_valid[0]
            );

            @(posedge clk);
            #1;

            $display(
                "        PE00: acc=%0d valid=%0b",
                $signed(c_output[0][0]),
                c_valid[0][0]
            );

        end

        // Allow pipeline to settle.
        repeat (4) begin
            @(posedge clk);
            #1;
        end

        // ------------------------------------------------
        // FINAL VERIFICATION
        // ------------------------------------------------

        for (i = 0; i < N; i = i + 1) begin
            for (j = 0; j < N; j = j + 1) begin

                if (c_valid[i][j] !== 1'b1) begin
                    $display(
                        "[FAIL] C[%0d][%0d] never became valid",
                        i,
                        j
                    );
                    $fatal;
                end

                if ($isunknown(c_output[i][j])) begin
                    $display(
                        "[FAIL] C[%0d][%0d] contains X/Z",
                        i,
                        j
                    );
                    $fatal;
                end

                if ($signed(c_output[i][j]) != expected[i][j]) begin
                    $display(
                        "[FAIL] C[%0d][%0d]: expected=%0d actual=%0d",
                        i,
                        j,
                        expected[i][j],
                        $signed(c_output[i][j])
                    );
                    $fatal;
                end

            end
        end

        $display("");
        $display("[PASS] %0dx%0d RTL systolic multiplication", N, N);
        $display("==============================================");

        for (i = 0; i < N; i = i + 1) begin
            $write("C[%0d] = ", i);

            for (j = 0; j < N; j = j + 1) begin
                $write("%0d ", $signed(c_output[i][j]));
            end

            $display("");
        end

        $display("==============================================");
        $display("ALL RTL SYSTOLIC TESTS PASSED");
        $display("==============================================");

        $finish;
    end

endmodule
