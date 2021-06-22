#include "mat_mul.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <omp.h>
#include <iostream>

#define MAX_GPU 4
#define YLEN 32

// below macro from 
// https://nval.andreasherten.de/2017/02/03/cuda-error-preprocessor-macros.html
#define CUDA_CALL( call )               \
  {                                     \
    cudaError_t result = call;          \
    if ( cudaSuccess != result ) {      \
      std::cerr << "CUDA error " << result << " in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString( result ) << " (" << #call << ")" << std::endl;  \
    } \
  }

static int num_threads = 8;

int num_devices = 0;

__global__ void sgemm(float* A, float* B, float* C, int M, int N, int K) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  if (i >= M || j >= N) return;

  float mid = 0;
  for (int k = 0; k < K; ++k) {
    mid += A[i * K + k] * B[k * N + j];
  }
  C[i * N + j] = mid;
}

static float *a_d[MAX_GPU], *b_d[MAX_GPU], *c_d[MAX_GPU];

void mat_mul(float *A, float *B, float *C, int M, int N, int K) {

#pragma omp parallel for num_threads(num_threads)
  for(int i=0;i<num_devices;i++) {
    CUDA_CALL( cudaSetDevice(i); )
    
    if(i!=num_devices-1) {
      CUDA_CALL( cudaMemcpy(a_d[i], A+M/num_devices*i * K, M/num_devices * K * sizeof(float), cudaMemcpyHostToDevice); )
    }
    else {
      CUDA_CALL( cudaMemcpy(a_d[i], A+M/num_devices*i * K, (M-M/num_devices*i) * K * sizeof(float), cudaMemcpyHostToDevice); )
    }
    CUDA_CALL( cudaMemcpy(b_d[i], B, K * N * sizeof(float), cudaMemcpyHostToDevice); )

    const int new_M = (M-M/num_devices*(num_devices-1));
    // Launch kernel
    dim3 blockDim(1, YLEN, 1);
    dim3 gridDim(new_M, N/YLEN+1, 1);
    sgemm<<<gridDim, blockDim>>>(a_d[i], b_d[i], c_d[i], new_M, N, K);

    if(i!=num_devices-1) {
      CUDA_CALL( cudaMemcpy(C+M/num_devices*i * N, c_d[i], M/num_devices * N * sizeof(float), cudaMemcpyDeviceToHost); )
    }
    else {
      CUDA_CALL( cudaMemcpy(C+M/num_devices*i * N, c_d[i], (M-M/num_devices*i) * N * sizeof(float), cudaMemcpyDeviceToHost); )
    }
  }

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  cudaDeviceSynchronize();
}

void mat_mul_init(float *A, float *B, float *C, int M, int N, int K) {

  CUDA_CALL( cudaGetDeviceCount(&num_devices); )

  printf("Using %d devices\n", num_devices);
  for (int i = 0; i < num_devices; i++) {
    cudaDeviceProp prop;
    CUDA_CALL( cudaGetDeviceProperties(&prop, i); )

    // Try printing more detailed information here
    printf("[GPU %d] %s\n", i, prop.name);
  }

  if ( num_devices <= 0 ) {
    printf("No CUDA device found. Aborting\n");
    exit(1);
  }

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL( cudaSetDevice(i); )

    const int new_M = (M-M/num_devices*(num_devices-1));
    // Allocate device memory
    CUDA_CALL( cudaMalloc(&a_d[i], new_M * K * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&b_d[i], K * N * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&c_d[i], new_M * N * sizeof(float)); )
  }
}

void mat_mul_final(float *A, float *B, float *C, int M, int N, int K) {
  // Do any post-matmul cleanup work here.
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL( cudaFree(a_d[i]); )
  }
}
