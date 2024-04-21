NVCC = nvcc

SOURCES = pso.cu
OBJECTS = pso.o
EXECUTABLE = pso

.PHONY: all clean detect_cuda_arch

all: detect_cuda_arch $(EXECUTABLE)
	@rm -f detect_cuda_arch cuda_arch.txt $(OBJECTS)

detect_cuda_arch: detect_cuda_arch.cu
	$(NVCC) $< -o $@
	./detect_cuda_arch > cuda_arch.txt
	@if [ $$? -ne 0 ]; then \
		echo "Error: Failed to detect CUDA architecture. Please check the detect_cuda_arch output for more details."; \
		exit 1; \
	fi

CUDA_ARCH = $(shell cat cuda_arch.txt 2>/dev/null)
CXXFLAGS = -std=c++17 -arch=$(CUDA_ARCH)
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart

$(EXECUTABLE): $(OBJECTS)
	$(NVCC) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

pso.o: pso.cu
	$(NVCC) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(EXECUTABLE)
