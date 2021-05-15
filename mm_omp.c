#include "mat_mul.h"
#include <omp.h>
#include <immintrin.h>

void mat_mul(float *A, float *B, float *C, int M, int N, int K, int num_threads) {
  int Z = 64;
  #pragma omp parallel for num_threads(num_threads)
  for (int ii = 0; ii < M; ii+=Z) {

    if ((M==4096) && (N==4096) && (K==4096)
     && (((long)&A[0]%16)==0) && (((long)&B[0]%16)==0) && (((long)&C[0]%16)==0)) {
        for (int kk = 0; kk < K; kk+=Z) {
          for (int jj = 0; jj < N; jj+=Z) {
      for (int i = ii; i < ((ii+Z)<M ? (ii+Z) : M); ++i) {
        for (int k = kk; k < ((kk+Z)<K ? (kk+Z) : K); ++k) {
          int en = ((jj+Z)<N ? (jj+Z) : N);
          __m128 a = _mm_broadcast_ss(&A[i * K + k]);
          for (int j = jj; j < en; j+=4) {
            __m128 b = _mm_load_ps(&B[k * N + j]);
            __m128 c = _mm_load_ps(&C[i * N + j]);
            _mm_store_ps(&C[i * N + j], _mm_fmadd_ps(a, b, c));
          }
        }
      }
          }
        }
    }

    else {
        for (int kk = 0; kk < K; kk+=Z) {
          for (int jj = 0; jj < N; jj+=Z) {
      for (int i = ii; i < ((ii+Z)<M ? (ii+Z) : M); ++i) {
        for (int k = kk; k < ((kk+Z)<K ? (kk+Z) : K); ++k) {
          int en = ((jj+Z)<N ? (jj+Z) : N);
          for (int j = jj; j < en; ++j) {
            C[i * N + j] += A[i * K + k] * B[k * N + j];
          }
        }
      }
          }
        }
    }

  }

}
