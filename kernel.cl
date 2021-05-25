#define BS (16*32)

__kernel void sgemm(__global float *A, __global float *B, __global float *C, int M, int N, int K) {
  int i = get_global_id(0); // row index of C
  int j = get_global_id(1); // column index of C
  if (i >= M || j >= N) return; // boundary check

  float res = 0;
  if (j<N/BS*BS) {
    int lj = get_local_id(1);
    __local float a_[BS];

    for (int bn = 0; bn < K/BS; bn++) {
      a_[lj] = A[i * K + (bn*BS+lj)];
      barrier(CLK_LOCAL_MEM_FENCE);

      for(int k=0;k<BS;k++) {
        res += a_[k] * B[(bn*BS+k) * N + j];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int k = K/BS*BS; k < K; k++) {
      res += A[i * K + k] * B[k * N + j];
    }
  }
  else {
    for (int k = 0; k < K; k++) {
      res += A[i * K + k] * B[k * N + j];
    }
  }
  C[i * N + j] = res;
}
