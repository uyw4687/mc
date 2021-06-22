#pragma once

#include <cstdio>
#include <cstdlib>

#ifdef USE_MPI

#include <mpi.h>

#define EXIT(status) \
  do { \
    MPI_Finalize(); \
    exit(status); \
  } while (0)

#define PRINTF_WITH_RANK(fmt, ...) \
  do { \
    printf("[rank %d] " fmt "\n", mpi_rank, ##__VA_ARGS__); \
  } while (0)

#define BARRIER() \
  do { \
    MPI_Barrier(MPI_COMM_WORLD); \
  } while (0)

#else

#define EXIT(status) \
  do { \
    exit(status); \
  } while (0)

#define PRINTF_WITH_RANK(fmt, ...) \
  do { \
    printf(fmt "\n", ##__VA_ARGS__); \
  } while (0)

#define BARRIER()

#endif

#define CHECK_ERROR(cond, fmt, ...) \
  do { \
    if (!(cond)) { \
      PRINTF_WITH_RANK("[%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
      EXIT(EXIT_FAILURE); \
    } \
  } while (false)

void* ReadFile(const char* filename, size_t* size);

void WriteFile(const char* filename, size_t size, void* buf);
