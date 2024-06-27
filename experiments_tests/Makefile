# Compiler
NVCC=nvcc

# Compiler flags
NVCCFLAGS=-O3

# Include directories
INCLUDES=

# Libraries
LIBS=

# Source files
SRCS=Stochastic_test.cpp

# Object files
OBJS=$(SRCS:.cpp=.o)
OBJS:=$(OBJS:.cu=.o)

# Executable name
EXEC=Stochastic_test

all: $(EXEC)

$(EXEC): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -x cu -dc $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -dc $< -o $@

clean:
	rm -f $(OBJS) $(EXEC)
