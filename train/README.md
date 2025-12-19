# Training & Benchmarking Guide

## Quick Start

### Benchmarks (with %)
```bash
cd P2 && make run    # P2: Direct Convolution
cd P3_1 && make run  # P3_1: Im2Col + GEMM (16x16)
cd P3_2 && make run  # P3_2: Optimized GEMM (32x32)
```

### Training
```bash
cd P2 && make train    # Train P2
cd P3_1 && make train  # Train P3_1
cd P3_2 && make train  # Train P3_2
```

### Inference
```bash
cd P2 && make infer    # Inference with P2
cd P3_1 && make infer  # Inference with P3_1
cd P3_2 && make infer  # Inference with P3_2
```

### Extract Features
```bash
cd P2 && make extract    # Extract with P2
cd P3_1 && make extract  # Extract with P3_1
cd P3_2 && make extract  # Extract with P3_2
```
