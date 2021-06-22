CXX=mpic++
TARGET=main
OBJECTS=timer.o util.o sep_mpi.o colorizer.o main.o

CXXFLAGS=-std=c++11 -O3 -Wall -fopenmp -I/usr/local/cuda/targets/x86_64-linux/include -DUSE_MPI
LDLIBS=-lm -L/usr/local/cuda/targets/x86_64-linux/lib -lm -lOpenCL -lcudart

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS) $(LDLIBS)

clean:
	rm -rf $(TARGET) $(OBJECTS)

colorizer.o: colorizer.cu
	nvcc -L/usr/local/lib -lmpi -I/usr/local/include -Xcompiler -fopenmp -c -o $@ $^

run: main
	sbatch run.sh input.bin ../common/network.bin output.bin
