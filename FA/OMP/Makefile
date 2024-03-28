CXX = g++
SOURCES = fa.cpp
OBJECTS = fa.o
EXECUTABLE = fa

.PHONY: all clean

all: $(EXECUTABLE)
	@rm -f $(OBJECTS)

CXXFLAGS = -std=c++17 -O3 -fopenmp

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

fa.o: fa.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(EXECUTABLE) $(OBJECTS)
