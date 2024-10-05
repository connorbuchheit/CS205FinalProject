# Compiler
CXX := mpic++
# Compiler flags
CXXFLAGS := -std=c++11 -Wall -fopenmp
# Include directories
INCLUDES := -I/shared/software/spack/opt/spack/linux-amzn2-skylake_avx512/gcc-7.3.1/eigen-3.3.7-pwe4bmtruhdeqccai3qasfyctdmb3qlj/include

# Source files
SRCS := icf.cpp parallel_ipm.cpp parallel_ipm_test.cpp utils.cpp
# Object files
OBJS := $(SRCS:.cpp=.o)
# Executable name
TARGET := ICFApp

# Makefile targets
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) *.err *.out
