CXX = g++
SOURCES = woa.cpp
OBJECTS = woa.o
EXECUTABLE = woa

.PHONY: all clean

all: $(EXECUTABLE)
	@rm -f $(OBJECTS)

CXXFLAGS = -std=c++17 -O3

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

pso.o: woa.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(EXECUTABLE) $(OBJECTS)
