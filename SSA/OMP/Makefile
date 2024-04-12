CXX = g++
SOURCES = ssa.cpp
OBJECTS = ssa.o
EXECUTABLE = ssa

.PHONY: all clean

all: $(EXECUTABLE)
	@rm -f $(OBJECTS)

CXXFLAGS = -std=c++17 -O3 -fopenmp

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

ssa.o: ssa.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(EXECUTABLE) $(OBJECTS)
