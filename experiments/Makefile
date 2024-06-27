# Compiler
NVCC=nvcc

# Compiler flags
NVCCFLAGS=-O3

# Include directories
INCLUDES=

# Libraries
LIBS=

# Source files
SRCS=MatrixMultiply.cpp

# Object files
OBJS=$(SRCS:.cpp=.o)
OBJS:=$(OBJS:.cu=.o)

# Executable name
EXEC=MatrixMultiply

all: $(EXEC)

$(EXEC): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -x cu -dc $< -o $@ -v

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -dc $< -o $@ -v

clean:
	rm -f $(OBJS) $(EXEC)
	rm -f *.csv
