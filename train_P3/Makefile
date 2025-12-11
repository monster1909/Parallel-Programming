# Auto-detect NVCC
NVCC := $(shell which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)

ifeq ($(shell test -x $(NVCC) && echo yes),yes)
    $(info Found NVCC at: $(NVCC))
else
    $(error NVCC not found. Please install CUDA Toolkit)
endif

CXXFLAGS := -O3 -std=c++17
INC := -I../phase3_gpu_optimized_v2/Include

SRC_CORE = ../phase3_gpu_optimized_v2/src/autoencoder.cu \
           ../phase3_gpu_optimized_v2/src/kernels/im2col.cu \
           ../phase3_gpu_optimized_v2/src/kernels/gemm.cu \
           ../phase3_gpu_optimized_v2/src/kernels/relu.cu \
           ../phase3_gpu_optimized_v2/src/kernels/maxpool.cu \
           ../phase3_gpu_optimized_v2/src/kernels/upsample.cu \
           ../phase3_gpu_optimized_v2/src/utils/cuda_utils.cpp \
           ../phase3_gpu_optimized_v2/src/utils/gpu_memory.cu

TARGET = train_p3

all: $(TARGET)

$(TARGET): train_p3.cu $(SRC_CORE)
	$(NVCC) $(CXXFLAGS) $(INC) $^ -o $@

clean:
	rm -f $(TARGET)

.PHONY: all clean

