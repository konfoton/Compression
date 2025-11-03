# Simple CUDA Makefile

NVCC ?= nvcc
ARCH ?= -gencode arch=compute_86,code=sm_86
CXXFLAGS ?= -O2 -std=c++14 $(ARCH)
INCLUDES := -Iinclude

SRC := \
	src/main.cu \
	src/kernels/runarray.cu

TARGET := build/app

.PHONY: all clean run

all: $(TARGET)


$(TARGET): $(SRC)
	@mkdir -p $(dir $@)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $@ $(SRC)

run: $(TARGET)
	$(TARGET)

clean:
	rm -rf build
