__kernel void NormalizeL(const __global float *input, __global float *output,
                         const int H, const int W) {
  const int h = get_global_id(0);
  const int w = get_global_id(1);

  output[h * W + w] = (input[h * W + w]-50)/100;
}

__kernel void Conv2d(const __global float *input, const __global float *weight, const __global float *bias, __global float *output,
                     const int stride, const int pad, const int dilation, const int has_bias_int,
                     const int C, const int H, const int W,
                     const int K, const int R, const int S,
                     const int OH, const int OW,
                     const int with_relu_int) {
  const int k = get_global_id(0);
  const int oh = get_global_id(1);
  const int ow = get_global_id(2);

  float o = (has_bias_int==1) ? bias[k] : 0;
  for (int c = 0; c < C; ++c) {
    for (int r = 0; r < R; ++r) {
      for (int s = 0; s < S; ++s) {
        int h = oh * stride - pad + r * dilation;
        int w = ow * stride - pad + s * dilation;
        if (h < 0 || h >= H || w < 0 || w >= W) continue;
        float i = input[c * H * W + h * W + w];
        float f = weight[k * C * R * S + c * R * S + r * S + s];
        o += i * f;
      }
    }
  }
  output[k * OH * OW + oh * OW + ow] = (with_relu_int==1) ? fmax(o,0) : o;
}

__kernel void BatchNorm2d(const __global float *input, const __global float *weight, const __global float *bias, const __global float *running_mean, const __global float *running_var, __global float *output,
                     const float eps,
                     const int C, const int H, const int W) {
  const int c = get_global_id(0);
  const int h = get_global_id(1);
  const int w = get_global_id(2);

  const int idx = c * H * W + h * W + w;
  output[idx] = (input[idx] - running_mean[c]) / sqrt(running_var[c] + eps) * weight[c] + bias[c];
}

__kernel void ConvTranspose2dReLU(const __global float *input, const __global float *weight, const __global float *bias, __global float *output,
                     const int stride, const int pad,
                     const int C, const int H, const int W,
                     const int K, const int R, const int S,
                     const int OH, const int OW) {
  const int k = get_global_id(0);
  const int oh = get_global_id(1);
  const int ow = get_global_id(2);

  float o = bias[k];
  for (int c = 0; c < C; ++c) {
    for (int r = 0; r < R; ++r) {
      for (int s = 0; s < S; ++s) {
        if ((oh + pad - r) % stride != 0) continue;
        if ((ow + pad - s) % stride != 0) continue;
        int h = (oh + pad - r) / stride;
        int w = (ow + pad - s) / stride;
        if (h < 0 || h >= H || w < 0 || w >= W) continue;
        float i = input[c * H * W + h * W + w];
        float f = weight[c * K * R * S + k * R * S + r * S + s];
        o += i * f;
      }
    }
  }
  output[k * OH * OW + oh * OW + ow] = fmax(o,0);
}

__kernel void Softmax(const __global float *input, __global float *output,
                      const int C, const int H, const int W) {
  const int h = get_global_id(0);
  const int w = get_global_id(1);

  float sum = 0;
  for(int c=0;c<C;++c) {
    sum += exp(input[c * H * W + h * W + w]);
  }
  for(int c=0;c<C;++c) {
    output[c * H * W + h * W + w] = exp(input[c * H * W + h * W + w]) / sum;
  }
}

__kernel void Upsample(const __global float *input, __global float *output,
                       const float scale_factor,
                       const int C, const int H, const int W, const int OH, const int OW) {
  const int c = get_global_id(0);
  const int oh = get_global_id(1);
  const int ow = get_global_id(2);

  const float h = (oh + 0.5) / scale_factor - 0.5;
  const float w = (ow + 0.5) / scale_factor - 0.5;
  int h0 = floor(h), w0 = floor(w);
  int h1 = h0 + 1, w1 = w0 + 1;
  const float h_offset = h - h0, w_offset = w - w0;
  const float w00 = (1 - h_offset) * (1 - w_offset);
  const float w01 = (1 - h_offset) * w_offset;
  const float w10 = h_offset * (1 - w_offset);
  const float w11 = h_offset * w_offset;
  h0 = h0 < 0 ? 0 : (h0 > H - 1 ? H - 1 : h0);
  h1 = h1 < 0 ? 0 : (h1 > H - 1 ? H - 1 : h1);
  w0 = w0 < 0 ? 0 : (w0 > W - 1 ? W - 1 : w0);
  w1 = w1 < 0 ? 0 : (w1 > W - 1 ? W - 1 : w1);
  output[c * OH * OW + oh * OW + ow] = w00 * input[c * H * W + h0 * W + w0]
                                     + w01 * input[c * H * W + h0 * W + w1]
                                     + w10 * input[c * H * W + h1 * W + w0]
                                     + w11 * input[c * H * W + h1 * W + w1];
}

__kernel void UnnormalizeAB(const __global float *input, __global float *output,
                            const int C, const int H, const int W) {
  const int c = get_global_id(0);
  const int h = get_global_id(1);
  const int w = get_global_id(2);

  output[c * H * W + h * W + w] = input[c * H * W + h * W + w]*110;
}
