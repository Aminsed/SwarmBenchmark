CXX = g++
SOURCES = goa.cpp
OBJECTS = goa.o
EXECUTABLE = goa

.PHONY: all clean

all: $(EXECUTABLE)
	@rm -f $(OBJECTS)

CXXFLAGS = -std=c++17 -O3 -fopenmp

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

goa.o: goa.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(EXECUTABLE) $(OBJECTS)
