#include "util.h"

#define NETW_SIZE 32241003

extern int mpi_size;

void _MPI_Bcast(int *N) {
  MPI_Bcast(N, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void __MPI_Bcast(float *network) {
  MPI_Bcast(network, NETW_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

void _MPI_Scatter(const void *sendbuf, int sendcount,
                  void *recvbuf, int recvcount, int root) {
  MPI_Scatter(sendbuf, sendcount, MPI_FLOAT, recvbuf, recvcount, MPI_FLOAT, root, MPI_COMM_WORLD);
}

void _MPI_Gather(const void *sendbuf, int sendcount, 
               void *recvbuf, int recvcount, int root) {
  MPI_Gather(sendbuf, sendcount, MPI_FLOAT, recvbuf, recvcount, MPI_FLOAT, root, MPI_COMM_WORLD);
}

void _MPI_Finalize() {
  MPI_Finalize();
}

