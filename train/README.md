# Training Module - Quick Guide

## Phase 2 (Basic GPU)

### Training
```bash
cd train/P2
make -f MAKEFILE
./train_phase2
```

### Inference
```bash
make -f Makefile.infer
./infer_phase2 weights/phase2_epoch_0.bin
```

### Feature Extraction
```bash
make -f Makefile.extract
./extract_features weights/phase2_epoch_0.bin features.bin
```

---

## Phase 3.1 (Im2Col + GEMM)

### Training
```bash
cd train/P3_1
make -f MAKEFILE
./train_phase3_v1
```

### Inference
```bash
make -f Makefile.infer
./infer_phase3_v1 weights/phase3_v1_epoch_0.bin
```

### Feature Extraction
```bash
make -f Makefile.extract
./extract_features weights/phase3_v1_epoch_0.bin features.bin
```

---

## Phase 3.2 (Optimized GEMM)

### Training
```bash
cd train/P3_2
make -f MAKEFILE
./train_phase3_v2
```

### Inference
```bash
make -f Makefile.infer
./infer_phase3_v2 weights/phase3_v2_epoch_0.bin
```

### Feature Extraction
```bash
make -f Makefile.extract
./extract_features weights/phase3_v2_epoch_0.bin features.bin
```

---

## Output Files

- **Weights**: `weights/phaseX_epoch_Y.bin` (~5 MB)
- **Features**: `features.bin` (~1.9 GB)
- **Logs**: `logs/training.log`

---

## Hyperparameters

Edit in training script (e.g., `train_phase2.cu` line 32-35):

```cpp
const int BATCH_SIZE = 32;
const int NUM_EPOCHS = 20;
const float LEARNING_RATE = 0.0001f;
```

Then rebuild: `make -f MAKEFILE clean && make -f MAKEFILE`
