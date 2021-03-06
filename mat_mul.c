#include "mat_mul.h"

#include <stdio.h>
#include <CL/cl.h>
#include <mpi.h>
#include "util.h"

#define CHECK_ERROR(err) \
  if (err != CL_SUCCESS) { \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE); \
  }

#define NUM_GPU 4

static cl_int err;
static cl_platform_id platform;
static cl_device_id devices[NUM_GPU];
static cl_context contexts[NUM_GPU];
static cl_command_queue queues[NUM_GPU];
static cl_program programs[NUM_GPU];
static cl_kernel kernels[NUM_GPU];
static cl_mem a_d[NUM_GPU], b_d[NUM_GPU], c_d[NUM_GPU];
static cl_uint num_dev;
static int num_threads = 4;

void ms(float *A, float *B, float *C, int M, int N, int K, int mpi_rank, int mpi_size) {
  if (mpi_rank == 0) {
    for (int i = 1; i < mpi_size-1; ++i) {
      MPI_Send(A+M/mpi_size*i*K, M/mpi_size * K, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
    }
    if (mpi_size != 1) {
      MPI_Send(A+M/mpi_size*(mpi_size-1)*K, M*K-M/mpi_size*(mpi_size-1) * K, MPI_FLOAT, mpi_size-1, 0, MPI_COMM_WORLD);
    }
  } else {
    if (mpi_rank!=mpi_size-1) {
      MPI_Recv(A, M/mpi_size * K, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, NULL);
    }
    else {
      MPI_Recv(A, M*K-M/mpi_size*mpi_rank * K, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, NULL);
    }
  }

  if (mpi_rank == 0) {
    for (int i = 1; i < mpi_size; ++i) {
      MPI_Send(B, K * N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(B, K * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, NULL);
  }
}

void mr(float *A, float *B, float *C, int M, int N, int K, int mpi_rank, int mpi_size) {
  if (mpi_rank == 0) {
    for (int i = 1; i < mpi_size-1; ++i) {
      MPI_Recv(C+M/mpi_size*i*N, M/mpi_size * N, MPI_FLOAT, i, 0, MPI_COMM_WORLD, NULL);
    }
    if (mpi_size != 1) {
      MPI_Recv(C+M/mpi_size*(mpi_size-1)*N, M*N-M/mpi_size*(mpi_size-1) * N, MPI_FLOAT, mpi_size-1, 0, MPI_COMM_WORLD, NULL);
    }
  } else {
    if (mpi_rank!=mpi_size-1) {
      MPI_Send(C, M/mpi_size * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
    else {
      MPI_Send(C, M*N-M/mpi_size*mpi_rank * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
  }
}

void mat_mul(float *A, float *B, float *C, int M, int N, int K, int mpi_rank, int mpi_size) {
  if (mpi_rank != 0) {
    alloc_mat(&A, M/mpi_size+(mpi_size-1), K);
    alloc_mat(&B, K, N);
    alloc_mat(&C, M/mpi_size+(mpi_size-1), N);
  }
  ms(A, B, C, M, N, K, mpi_rank, mpi_size);

  int new_M = M/mpi_size;
  if (mpi_rank==mpi_size-1) {
    new_M = M/mpi_size+mpi_size-1;
  }

#pragma omp parallel for num_threads(num_threads)
  for (int i = 0;i < num_dev; i++) {
    if (i!=num_dev-1) {
      err = clEnqueueWriteBuffer(queues[i], a_d[i], CL_TRUE, 0, new_M/num_dev * K * sizeof(float), A+new_M/num_dev*i*K, 0, NULL, NULL);
    }
    else {
      err = clEnqueueWriteBuffer(queues[i], a_d[i], CL_TRUE, 0, new_M*K*sizeof(float)-new_M/num_dev*i*K * sizeof(float), A+new_M/num_dev*i*K, 0, NULL, NULL);
    }
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queues[i], b_d[i], CL_TRUE, 0, K * N * sizeof(float), B, 0, NULL, NULL);
    CHECK_ERROR(err);
  }

#pragma omp parallel for num_threads(num_threads)
  for (int i = 0;i < num_dev; i++) {
    err = clSetKernelArg(kernels[i], 0, sizeof(cl_mem), &a_d[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernels[i], 1, sizeof(cl_mem), &b_d[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernels[i], 2, sizeof(cl_mem), &c_d[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernels[i], 3, sizeof(int), &M);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernels[i], 4, sizeof(int), &N);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernels[i], 5, sizeof(int), &K);
    CHECK_ERROR(err);
  }

  size_t gws[2] = {new_M/num_dev+num_dev-1, N}, lws[2] = {1, 16*32};
  for (int i = 0; i < 2; ++i) {
    gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
  }

#pragma omp parallel for num_threads(num_threads)
  for (int i = 0;i < num_dev; i++) {
    err = clEnqueueNDRangeKernel(queues[i], kernels[i], 2, NULL, gws, lws, 0, NULL, NULL);
    CHECK_ERROR(err);
  }

#pragma omp parallel for num_threads(num_threads)
  for (int i = 0;i < num_dev; i++) {
    if (i!=num_dev-1) {
      err = clEnqueueReadBuffer(queues[i], c_d[i], CL_TRUE, 0, new_M/num_dev * N * sizeof(float), C+new_M/num_dev*i*N, 0, NULL, NULL);
    }
    else {
      err = clEnqueueReadBuffer(queues[i], c_d[i], CL_TRUE, 0, new_M * N * sizeof(float) - new_M/num_dev*i*N*sizeof(float), C+new_M/num_dev*i*N, 0, NULL, NULL);
    }
    CHECK_ERROR(err);
  }
  mr(A, B, C, M, N, K, mpi_rank, mpi_size);

  if (mpi_rank != 0) {
    free(A);
    free(B);
    free(C);
  }
}

static void print_platform_info(cl_platform_id platform) {
  size_t sz;
  char *buf;
  CHECK_ERROR(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &sz));
  buf = (char*)malloc(sz);
  CHECK_ERROR(clGetPlatformInfo(platform, CL_PLATFORM_NAME, sz, buf, NULL));
  printf("Detected OpenCL platform: %s\n", buf);
  free(buf);
}

static void print_device_info(cl_device_id device) {
  size_t sz;
  char *buf;
  CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &sz));
  buf = (char*)malloc(sz);
  CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_NAME, sz, buf, NULL));
  printf("Detected OpenCL device: %s\n", buf);
  free(buf);
}

static cl_program create_and_build_program_with_source(cl_context context, cl_device_id device, const char *file_name) {
  FILE *file = fopen(file_name, "rb");
  if (file == NULL) {
    printf("Failed to open %s\n", file_name);
    exit(EXIT_FAILURE);
  }
  fseek(file, 0, SEEK_END);
  size_t source_size = ftell(file);
  rewind(file);
  char *source_code = (char*)malloc(source_size + 1);
  fread(source_code, sizeof(char), source_size, file);
  source_code[source_size] = '\0';
  fclose(file);
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_code, &source_size, &err);
  CHECK_ERROR(err);
  free(source_code);
  err = clBuildProgram(program, 1, &device, "", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    CHECK_ERROR(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
    char *log = (char*)malloc(log_size + 1);
    CHECK_ERROR(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL));
    log[log_size] = 0;
    printf("Compile error:\n%s\n", log);
    free(log);
  }
  CHECK_ERROR(err);
  return program;
}

void mat_mul_init(float *A, float *B, float *C, int M, int N, int K, int mpi_rank, int mpi_size) {
  err = clGetPlatformIDs(1, &platform, NULL);
  CHECK_ERROR(err);
  printf("[rank %d] ", mpi_rank);
  print_platform_info(platform);

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, NUM_GPU, devices, &num_dev);
  CHECK_ERROR(err);
  if (num_dev>NUM_GPU) {
    num_dev = NUM_GPU;
  }
  for (int i = 0;i < num_dev; i++) {
    printf("[rank %d] Devide %d: ", mpi_rank, i);
    print_device_info(devices[i]);
  }

  for (int i = 0;i < num_dev; i++) {
    contexts[i] = clCreateContext(NULL, 1, &devices[i], NULL, NULL, &err);
    CHECK_ERROR(err);
  }

  for (int i = 0;i < num_dev; i++) {
    queues[i] = clCreateCommandQueue(contexts[i], devices[i], 0, &err);
    CHECK_ERROR(err);
  }

  for (int i = 0;i < num_dev; i++) {
    programs[i] = create_and_build_program_with_source(contexts[i], devices[i], "kernel.cl");
  }

  for (int i = 0;i < num_dev; i++) {
    kernels[i] = clCreateKernel(programs[i], "sgemm", &err); 
    CHECK_ERROR(err);
  }

  int new_M = M/mpi_size;
  if (mpi_rank==mpi_size-1) {
    new_M = M/mpi_size+mpi_size-1;
  }

  for (int i = 0;i < num_dev; i++) {
    a_d[i] = clCreateBuffer(contexts[i], CL_MEM_READ_WRITE, (new_M/num_dev+num_dev-1) * K * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    b_d[i] = clCreateBuffer(contexts[i], CL_MEM_READ_WRITE, K * N * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    c_d[i] = clCreateBuffer(contexts[i], CL_MEM_READ_WRITE, (new_M/num_dev+num_dev-1) * N * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
  }

  for (int i = 0;i < num_dev; i++) {
    err = clFinish(queues[i]);
    CHECK_ERROR(err);
  }
}

void mat_mul_final(float *A, float *B, float *C, int M, int N, int K, int mpi_rank, int mpi_size) {
  for (int i = 0;i < num_dev; i++) {
    err = clFinish(queues[i]);
    CHECK_ERROR(err);
  }
}
