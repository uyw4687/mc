__kernel void sgemm(__global float *A, __global float *B, __global float *C, int M, int N, int K) {
  int i = get_global_id(0);
  int j = get_global_id(1);
  if (i >= M || j >= N) return;
  float result = 0;
  for (int k = 0; k < K; k++) {
    result += A[i * K + k] * B[k * N + j];
  }
  C[i * N + j] = result;
}
