CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -march=native -pthread
TARGET_GENOMIC = morans_i_genomic
SOURCES_GENOMIC = main_genomic.cpp morans_i_genomic.cpp
OBJECTS_GENOMIC = $(SOURCES_GENOMIC:.cpp=.o)

# Default target
all: $(TARGET_GENOMIC)


# Build the genomic executable
$(TARGET_GENOMIC): $(OBJECTS_GENOMIC)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Build object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -f $(OBJECTS_GENOMIC) $(TARGET_GENOMIC)


# Run the genomic program
run-genomic: $(TARGET_GENOMIC)
	./$(TARGET_GENOMIC)

# Debug build
debug: CXXFLAGS = -std=c++17 -O0 -g -Wall -Wextra -pthread -DDEBUG
debug: $(TARGET_GENOMIC)

.PHONY: all clean run-genomic debug