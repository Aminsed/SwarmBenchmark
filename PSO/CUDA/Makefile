NVCC = nvcc
SRC_DIR = src
INCLUDE_DIR = include
OBJ_DIR = obj
BIN_DIR = bin

SOURCES = $(wildcard $(SRC_DIR)/*.cu)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
EXECUTABLE = $(BIN_DIR)/pso

.PHONY: all clean detect_cuda_arch

all: detect_cuda_arch $(EXECUTABLE)

detect_cuda_arch: detect_cuda_arch.cu
	$(NVCC) $< -o $@
	./detect_cuda_arch > cuda_arch.txt
	@if [ $$? -ne 0 ]; then \
		echo "Error: Failed to detect CUDA architecture. Please check the detect_cuda_arch output for more details."; \
		exit 1; \
	fi

CUDA_ARCH = $(shell cat cuda_arch.txt 2>/dev/null)

CXXFLAGS = -std=c++17 -Iinclude -arch=$(CUDA_ARCH)
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart

$(EXECUTABLE): $(OBJECTS) | $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(CXXFLAGS) -c $< -o $@

$(BIN_DIR) $(OBJ_DIR):
	mkdir -p $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) detect_cuda_arch cuda_arch.txt
