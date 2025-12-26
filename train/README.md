# CIFAR-10 Autoencoder Training

## ⚠️ Important
**Must `cd` into P2/P3_1/P3_2 before running commands!**

## Commands

### Benchmark
```bash
cd P2 && make -f MAKEFILE run
cd P3_1 && make -f MAKEFILE run  
cd P3_2 && make -f MAKEFILE run
```

### Training
```bash
cd P2 && make -f MAKEFILE train
cd P3_1 && make -f MAKEFILE train
cd P3_2 && make -f MAKEFILE train
```

### Inference
```bash
# Using Makefile (uses default weights)
cd P2 && make -f MAKEFILE infer
cd P3_1 && make -f MAKEFILE infer
cd P3_2 && make -f MAKEFILE infer

# Manual with specific weights
./infer_phase2 weights/phase2_epoch_5.bin
./infer_phase3_v1 weights/phase3_v1_best.bin
./infer_phase3_v2 weights/phase3_v2_epoch_10.bin
```

### Feature Extraction
```bash
# Using Makefile (uses default weights)
cd P2 && make -f MAKEFILE extract
cd P3_1 && make -f MAKEFILE extract
cd P3_2 && make -f MAKEFILE extract

# Manual with specific weights
./extract_features weights/phase2_epoch_5.bin features.bin
./extract_features weights/phase3_v1_best.bin features.bin
./extract_features weights/phase3_v2_epoch_10.bin features.bin
```

## Makefile Targets

| Target | Description |
|--------|-------------|
| `clean` | Remove build files |
| `all` | Build all programs |
| `run` | Run benchmark |
| `train` | Run training |
| `infer` | Run inference |
| `extract` | Extract features |

