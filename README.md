# CIFAR-10 Autoencoder - GPU Optimization Project

## Overview
Complete implementation of CIFAR-10 autoencoder with progressive GPU optimization, from CPU baseline to optimized GPU kernels, plus SVM classifier for latent features.

## Project Structure

```
├── phase1_cpu_baseline/        # CPU baseline with OpenMP
├── phase2_gpu_basic/           # GPU with direct convolution
├── phase3_gpu_optimized/       # Im2Col + GEMM (16×16 tiling)
├── phase3_gpu_optimized_v2/    # Optimized GEMM (32×32 tiling)
├── phase4_svm/                 # SVM classifier (CPU)
├── phase4_svm_gpu/             # SVM classifier (GPU)
├── train/                      # Training & inference programs
│   ├── P2/                     # Phase 2 training
│   ├── P3_1/                   # Phase 3 v1 training
│   └── P3_2/                   # Phase 3 v2 training
└── Data/                       # CIFAR-10 dataset
```

## Requirements
- **GPU**: NVIDIA GPU with CUDA support (GTX 1650+)
- **CUDA**: Version 12.6+
- **Compiler**: g++ with C++17, nvcc
- **Dataset**: CIFAR-10 binary format in `Data/cifar-10-batches-bin/`

## Architecture

```
Input: 32×32×3
├─ Encoder:
│  ├─ Conv1: 3→256, ReLU, MaxPool(2×2) → 256×16×16
│  └─ Conv2: 256→128, ReLU, MaxPool(2×2) → 128×8×8
├─ Latent Space: 128×8×8 (8,192 dimensions)
└─ Decoder:
   ├─ DeConv1: 128→128, Upsample(2×) → 128×16×16
   ├─ DeConv2: 128→256, Upsample(2×) → 256×32×32
   └─ FinalConv: 256→3 → 3×32×32
Output: 32×32×3 (reconstructed)
```

## Quick Start

### 1. Benchmark Performance
Compare different optimization phases:

```bash
# Phase 1: CPU Baseline
cd phase1_cpu_baseline && make && ./autoencoder info

# Phase 2: GPU Basic
cd phase2_gpu_basic && make -f MAKEFILE
./test_gpu      # Single image
./run_gpu       # 60k images

# Phase 3: GPU Optimized
cd phase3_gpu_optimized && make -f MAKEFILE
./test_gpu      # Single image
./run_phase3    # 60k images

# Phase 3 V2: Highly Optimized
cd phase3_gpu_optimized_v2 && make -f MAKEFILE
./test_gpu          # Single image
./run_phase3        # 60k images
./feature_extract   # Encoder only
```

### 2. Training
Train autoencoder models (see `train/README.md` for details):

```bash
cd train/P3_1
make -f MAKEFILE train
./train_phase3_v1
```

### 3. Inference
Run inference with trained weights:

```bash
cd train/P3_1
make -f MAKEFILE infer
./infer_phase3_v1 weights/phase3_v1_best.bin
```

### 4. Feature Extraction
Extract latent features for classification:

```bash
cd train/P3_1
make -f MAKEFILE extract
./extract_features weights/phase3_v1_best.bin features.bin
```

### 5. SVM Classification
Train and test SVM on extracted features:

```bash
# Train SVM
cd phase4_svm
make
./train_svm train_features.bin model.svm

# Test SVM
./test_svm model.svm test_features.bin results.csv
```

## Optimization Techniques

| Phase | Technique | Key Features |
|-------|-----------|--------------|
| **Phase 1** | CPU + OpenMP | Baseline, im2col + GEMM, parallel batches |
| **Phase 2** | GPU Direct Conv | Direct convolution kernels on GPU |
| **Phase 3** | Im2Col + GEMM | Tiled GEMM (16×16), shared memory |
| **Phase 3 V2** | Optimized GEMM | Larger tiles (32×32), batching, kernel fusion |

## Performance Comparison

Expected speedup (60k images):
- **Phase 1 (CPU)**: Baseline
- **Phase 2 (GPU)**: ~10-20× faster than CPU
- **Phase 3**: ~2-3× faster than Phase 2
- **Phase 3 V2**: ~1.5-2× faster than Phase 3

## Detailed Documentation

- **Training & Inference**: See `train/README.md`
- **CPU Baseline**: See `phase1_cpu_baseline/readme.md`
- **SVM Classifier**: See `phase4_svm/readme.md`

## Workflow

1. **Extract features** using trained autoencoder encoder
2. **Train SVM** on extracted latent features (8,192 dims)
3. **Classify** CIFAR-10 images using SVM on latent space

## Notes

- All GPU programs require CUDA-capable GPU
- Training produces `.bin` weight files in `weights/` directory
- Feature extraction outputs binary files with format: `[label:1byte][features:8192×float]`
- SVM expects feature files from extraction step

