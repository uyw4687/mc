#define BLK_CHN 64
#define WLEN 2

__kernel void NormalizeL(const __global float *input, __global float *output) {
  const int H = 256;
  const int W = 256;

  const int h = get_global_id(0);
  const int w = get_global_id(1)*WLEN;
  if (h >= H || w >= W) return; 

  const int hW_w = h * W + w;
  for(int j=0;j<WLEN;j++) {
    output[hW_w+j] = input[hW_w+j]/100-0.5;
  }
}

__kernel void Conv2d(const __global float *input, const __global float *weight, const __global float *bias, __global float *output,
                     const int stride, const int pad, const int dilation, const int has_bias_int,
                     const int C, const int H,
                     const int K, const int R,
                     const int OH,
                     const int with_relu_int) {
  const int W = H;
  const int S = R;
  const int OW = OH;

  const int k = get_global_id(0);
  const int oh = get_global_id(1);
  const int ow = get_global_id(2)*WLEN;
  if (k >= K || oh >= OH || ow >= OW) return; 

  const int RS = R * S;
  const int kCRS = k * C * RS;
  
  float o[WLEN] = {[0 ... WLEN-1] = (has_bias_int==1) ? bias[k] : 0};

  for (int c = 0; c < C; ++c) {
    const int kCRS_cRS = kCRS + c * RS;
    const int cHW = c * H * W;
    for (int r = 0; r < R; ++r) {
      const int kCRS_cRS_rS = kCRS_cRS + r * S;
      for (int s = 0; s < S; ++s) {
        const int h = oh * stride - pad + r * dilation;
        int w[WLEN];
        w[0] = ow * stride - pad + s * dilation;
        for(int j=1;j<WLEN;j++) {
          w[j] = w[j-1] + stride;
        }

        const float wt = weight[kCRS_cRS_rS + s];
        const int cHW_hW = cHW + h * W;
        for (int j=0;j<WLEN;j++) {
          if (h < 0 || h >= H || w[j] < 0 || w[j] >= W) {}
          else {
            o[j] += input[cHW_hW + w[j]] * wt;
          }
        }
      }
    }
  }
  const int kOHOW_ohOW_ow =  k * OH * OW + oh * OW + ow;
  for (int j=0;j<WLEN;j++) {
    output[kOHOW_ohOW_ow+j] = (with_relu_int==1) ? fmax(o[j],0) : o[j];
  }
}

__kernel void Conv2d64(const __global float *input, const __global float *weight, const __global float *bias, __global float *output,
                     const int stride, const int pad, const int dilation,
                     const int C, const int H,
                     const int K,
                     const int OH) {
  const int W = H;
  const int R = 3;
  const int S = 3;
  const int OW = OH;

  const int k = get_global_id(0);
  const int oh = get_global_id(1);
  const int ow = get_global_id(2)*WLEN;
  if (k >= K || oh >= OH || ow >= OW) return; 

  const int blen_1 = get_local_size(1);
  const int blen_2 = get_local_size(2);
  const int index_flattened = get_local_id(1)*blen_2 + get_local_id(2);

  const int DIV = C/BLK_CHN;
  const int ch_div = BLK_CHN/blen_1/blen_2;
  const int RS = R * S;
  const int kCRS = k * C * RS;

  float o[WLEN] = {[0 ... WLEN-1] = bias[k]};
  __local float weight_local[BLK_CHN*3*3];

  for (int i=0;i<DIV;i++) {
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int c = ch_div*index_flattened; c < ch_div*(index_flattened+1); ++c) {
      const int kCRS_cRS = kCRS + (c + i*BLK_CHN) * RS;
      const int _cRS = c * RS;
      for (int r = 0; r < R; ++r) {
        const int rS = r * S;
        const int _cRS_rS = _cRS + rS;
        for (int s = 0; s < S; ++s) {
          weight_local[_cRS_rS + s] = weight[kCRS_cRS + rS + s];
        }
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int c = i*BLK_CHN; c < (i+1)*BLK_CHN; ++c) {
      const int cHW = c * H * W;
      const int _cRS = (c-i*BLK_CHN) * RS;
      for (int r = 0; r < R; ++r) {
        const int _cRS_rS = _cRS + r * S;
        for (int s = 0; s < S; ++s) {
          const int h = oh * stride - pad + r * dilation;
          const int cHW_hW = cHW + h * W;
          const float wt = weight_local[_cRS_rS + s];
          int w[WLEN];

          w[0] = ow * stride - pad + s * dilation;
          for(int j=1;j<WLEN;j++) {
            w[j] = w[j-1] + stride;
          }

          for (int j=0;j<WLEN;j++) {
            if (h >= 0 && h < H && w[j] >= 0 && w[j] < W) {
              o[j] += input[cHW_hW + w[j]] * wt;
            }
          }
        }
      }
    }
  }
  
  const int ind = k * OH * OW + oh * OW + ow;
  for (int j=0;j<WLEN;j++) {
    output[ind+j] = fmax(o[j],0);
  }
}

__kernel void BatchNorm2d(const __global float *input, const __global float *weight, const __global float *bias, const __global float *running_mean, const __global float *running_var, __global float *output,
                     const int C, const int H) {
  const float eps = 1e-5;
  const int W = H;
  const int c = get_global_id(0);
  const int h = get_global_id(1);
  const int w = get_global_id(2);
  if (c >= C || h >= H || w >= W) return; 

  const int cHW_hW_w = c * H * W + h * W + w;
  output[cHW_hW_w] = (input[cHW_hW_w] - running_mean[c]) / sqrt(running_var[c] + eps) * weight[c] + bias[c];
}

__kernel void ConvTranspose2dReLU(const __global float *input, const __global float *weight, const __global float *bias, __global float *output) {
  const int stride = 2;
  const int pad = 1;
  const int C = 512;
  const int H = 32;
  const int W = 32;
  const int K = 256;
  const int R = 4;
  const int S = 4;
  const int OH = 64;
  const int OW = 64;

  const int k = get_global_id(0);
  const int oh = get_global_id(1);
  const int ow = get_global_id(2);
  if (k >= K || oh >= OH || ow >= OW) return; 

  const int blen_2 = get_local_size(2);
  const int index_flattened = get_local_id(2);

  const int DIV = C/BLK_CHN;
  const int ch_div = BLK_CHN/blen_2;
  const int RS = R * S;
  const int HW = H * W;
  const int KRS = K * RS;
  const int kRS = k * RS;

  float o = bias[k];
  __local float weight_local[BLK_CHN*4*4];

  for (int i=0;i<DIV;i++) {
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int c = ch_div*index_flattened; c < ch_div*(index_flattened+1); ++c) {
      const int kRS_cKRS = kRS + (c + i*BLK_CHN) * KRS;
      const int _cRS = c * RS;
      for (int r = 0; r < R; ++r) {
        const int rS = r * S;
        const int _cRS_rS = _cRS + rS;
        for (int s = 0; s < S; ++s) {
          weight_local[_cRS_rS + s] = weight[kRS_cKRS + rS + s];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int c = i*BLK_CHN; c < (i+1)*BLK_CHN; ++c) {
      const int _cRS = (c-i*BLK_CHN) * RS;
      const int cHW = c * HW;
      for (int r = 0; r < R; ++r) {
        const int _cRS_rS = _cRS + r * S;
        for (int s = 0; s < S; ++s) {
          const int oh_pad__r = oh + pad - r;
          const int ow_pad__s = ow + pad - s;
          const int h = oh_pad__r / stride;
          const int w = ow_pad__s / stride;
          if ((oh_pad__r % stride != 0) || (ow_pad__s % stride != 0) ||
               (h < 0 || h >= H || w < 0 || w >= W)) continue;
          o += input[cHW + h * W + w] * weight_local[_cRS_rS + s];
        }
      }
    }
  }

  output[k * OH * OW + oh * OW + ow] = fmax(o,0);
}

__kernel void Softmax(const __global float *input, __global float *output) {
  const int C = 313;
  const int H = 64;
  const int W = 64;

  const int h = get_global_id(0);
  const int w = get_global_id(1);
  if (h >= H || w >= W) return; 

  const int HW = H * W;
  const int hW_w = h * W + w;

  float exp_input_reg[313];
  float sum = 0;

  for(int c=0;c<C;++c) {
    exp_input_reg[c] = exp(input[c * HW + hW_w]);
    sum += exp_input_reg[c];
  }
  for(int c=0;c<C;++c) {
    output[c * HW + hW_w] = exp_input_reg[c] / sum;
  }
}

__kernel void UpsampleUnnormalize(const __global float *input, __global float *output) {
  const float scale_factor = 4;
  const int C = 2;
  const int H = 64;
  const int W = 64;
  const int OH = 256;
  const int OW = 256;
 
  const int c = get_global_id(0);
  const int oh = get_global_id(1);
  const int ow = get_global_id(2);
  if (c >= C || oh >= OH || ow >= OW) return; 

  const float miv = 0.5 / scale_factor - 0.5;
  const float h = oh / scale_factor + miv;
  const float w = ow / scale_factor + miv;
  int h0 = floor(h), w0 = floor(w);
  int h1 = h0 + 1, w1 = w0 + 1;

  const float h_offset = h - h0, w_offset = w - w0;
  const float om_ho = 1 - h_offset;
  const float om_wo = 1 - w_offset;
  const float w00 = om_ho * om_wo;
  const float w01 = om_ho * w_offset;
  const float w10 = h_offset * om_wo;
  const float w11 = h_offset * w_offset;

  const int hm1 = H - 1;
  const int wm1 = W - 1;
  h0 = h0 < 0 ? 0 : (h0 > hm1 ? hm1 : h0);
  h1 = h1 < 0 ? 0 : (h1 > hm1 ? hm1 : h1);
  w0 = w0 < 0 ? 0 : (w0 > wm1 ? wm1 : w0);
  w1 = w1 < 0 ? 0 : (w1 > wm1 ? wm1 : w1);
  
  const int cHW = c * H * W;
  const int h0W = h0 * W;
  const int h1W = h1 * W;
  const int cHW_h0W = cHW + h0W;
  const int cHW_h1W = cHW + h1W;
  float mid = w00 * input[cHW_h0W + w0]
            + w01 * input[cHW_h0W + w1]
            + w10 * input[cHW_h1W + w0]
            + w11 * input[cHW_h1W + w1];

  output[c * OH * OW + oh * OW + ow] = mid*110;
}

