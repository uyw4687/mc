void _MPI_Bcast(int *N);
void __MPI_Bcast(float *N);
void _MPI_Scatter(const void *sendbuf, int sendcount,
                  void *recvbuf, int recvcount, int root);
void _MPI_Gather(const void *sendbuf, int sendcount, 
               void *recvbuf, int recvcount, int root);
void _MPI_Finalize();

