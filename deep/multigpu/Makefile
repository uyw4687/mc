TARGET=main
OBJECTS=timer.o util.o colorizer.o

CXXFLAGS=-std=c++11 -O3 -Wall -fopenmp -I/usr/local/cuda/targets/x86_64-linux/include
LDLIBS=-lm -L/usr/local/cuda/targets/x86_64-linux/lib -lm -lOpenCL

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)

run: main
	sbatch run.sh input.bin ../common/network.bin output.bin
