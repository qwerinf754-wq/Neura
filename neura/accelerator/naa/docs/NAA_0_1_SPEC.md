# NAA-0.1 Specification

## Purpose

Neura AI Accelerator prototype for Transformer workloads.

## Compute

- MAC units: 256
- Matrix tile: 16 x 16
- Target clock: 200 MHz
- Peak MAC rate: 51.2 GMAC/s
- Peak FLOP rate: 102.4 GFLOP/s

## Precision

- BF16: planned
- FP16: planned
- FP32 accumulation: planned
- INT8: not in v0.1

## Memory

- Local SRAM target: 1 MiB
- SRAM banks: 16
- Bank size: 64 KiB
- Vector width: 256 bits

## Commands

Instruction width: 128 bits.

Opcode:
- 0x01 NOP
- 0x02 LOAD
- 0x03 STORE
- 0x04 GEMM
- 0x05 GEMV
- 0x06 VECTOR
- 0x07 RMSNORM
- 0x08 ROPE
- 0x09 SWIGLU
- 0x0A WAIT
- 0x0B BARRIER
- 0x0C HALT

## Development stages

1. MAC unit
2. MAC array
3. GEMM
4. BF16
5. FP16
6. SRAM
7. DMA
8. Command queue
9. Vector engine
10. Neura runtime
11. FPGA prototype
