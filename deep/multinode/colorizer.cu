#include "colorizer.h"

#include <cmath>
#include <iostream>
#include <CL/cl.h>

#include "util.h"
#include "timer.h"
#include "sep_mpi.h"

#include "kernels.cu"

#define CHK_LEN 32
#define NUM_GPU 4
#define WLEN 1
#define WLEN_N 2

#define NETW_SIZE 32241003
#define CHK_DIV 30
#define CHK_LEFT 2

#define CHECK_ERROR_CL(err) \
  if (err != CL_SUCCESS) { \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE); \
  }
  
// below macro from 
// https://nval.andreasherten.de/2017/02/03/cuda-error-preprocessor-macros.html
#define CUDA_CALL( call )               \
  {                                     \
    cudaError_t result = call;          \
    if ( cudaSuccess != result ) {      \
      std::cerr << "CUDA error " << result << " in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString( result ) << " (" << #call << ")" << std::endl;  \
    } \
  }

/*
 * Declarations
 */

struct Tensor {
  // Pointer to data
  float* buf;
  // Shape of tensor, from outermost dimension to innermost dimension.
  // e.g., {{1.0, -0.5, 2.3}, {4.3, 5.6, -7.8}} => shape = {2, 3}
  int shape[4];
};

extern int mpi_rank, mpi_size;

typedef struct Features {
  Tensor fm_normalize_l,
         fm1_0, fm1_1, fm1_2,
         fm2_0, fm2_1, fm2_2,
         fm3_0, fm3_1, fm3_2, fm3_3,
         fm4_0, fm4_1, fm4_2, fm4_3,
         fm5_0, fm5_1, fm5_2, fm5_3,
         fm6_0, fm6_1, fm6_2, fm6_3,
         fm7_0, fm7_1, fm7_2, fm7_3,
         fm8_0, fm8_1, fm8_2, fm8_3,
         fm_softmax, fm_model_out, fm_upsample4;
} Features;
// Array of Feature maps
static Features fss[CHK_LEN];

typedef struct Weights {
  Tensor *model1_0_weight, *model1_0_bias, *model1_2_weight, *model1_2_bias, *model1_4_weight, *model1_4_bias, 
         *model2_0_weight, *model2_0_bias, *model2_2_weight, *model2_2_bias, *model2_4_weight, *model2_4_bias, 
         *model3_0_weight, *model3_0_bias, *model3_2_weight, *model3_2_bias, *model3_4_weight, *model3_4_bias, *model3_6_weight, *model3_6_bias,
         *model4_0_weight, *model4_0_bias, *model4_2_weight, *model4_2_bias, *model4_4_weight, *model4_4_bias, *model4_6_weight, *model4_6_bias,
         *model5_0_weight, *model5_0_bias, *model5_2_weight, *model5_2_bias, *model5_4_weight, *model5_4_bias, *model5_6_weight, *model5_6_bias,
         *model6_0_weight, *model6_0_bias, *model6_2_weight, *model6_2_bias, *model6_4_weight, *model6_4_bias, *model6_6_weight, *model6_6_bias,
         *model7_0_weight, *model7_0_bias, *model7_2_weight, *model7_2_bias, *model7_4_weight, *model7_4_bias, *model7_6_weight, *model7_6_bias,
         *model8_0_weight, *model8_0_bias, *model8_2_weight, *model8_2_bias, *model8_4_weight, *model8_4_bias, *model8_6_weight, *model8_6_bias,
         *model_out_weight,
         *model1_4_running_mean, *model1_4_running_var,
         *model2_4_running_mean, *model2_4_running_var,
         *model3_6_running_mean, *model3_6_running_var,
         *model4_6_running_mean, *model4_6_running_var,
         *model5_6_running_mean, *model5_6_running_var,
         *model6_6_running_mean, *model6_6_running_var,
         *model7_6_running_mean, *model7_6_running_var;
} Weights;

// cuda
static int num_devices = 0;

typedef struct Features_cud {
  float *image_L_d, *image_AB_d,
        *fm_normalize_l_d,
        *fm1_0_d, *fm1_1_d, *fm1_2_d,
        *fm2_0_d, *fm2_1_d, *fm2_2_d,
        *fm3_0_d, *fm3_1_d, *fm3_2_d, *fm3_3_d,
        *fm4_0_d, *fm4_1_d, *fm4_2_d, *fm4_3_d,
        *fm5_0_d, *fm5_1_d, *fm5_2_d, *fm5_3_d,
        *fm6_0_d, *fm6_1_d, *fm6_2_d, *fm6_3_d,
        *fm7_0_d, *fm7_1_d, *fm7_2_d, *fm7_3_d,
        *fm8_0_d, *fm8_1_d, *fm8_2_d, *fm8_3_d,
        *fm_softmax_d, *fm_model_out_d, *fm_upsample4_d;
} Features_cud;
static Features_cud fscs[CHK_LEN];
typedef struct Weights_cud {
  float *model1_0_weight_d, *model1_0_bias_d, *model1_2_weight_d, *model1_2_bias_d, *model1_4_weight_d, *model1_4_bias_d,
         *model2_0_weight_d, *model2_0_bias_d, *model2_2_weight_d, *model2_2_bias_d, *model2_4_weight_d, *model2_4_bias_d,
         *model3_0_weight_d, *model3_0_bias_d, *model3_2_weight_d, *model3_2_bias_d, *model3_4_weight_d, *model3_4_bias_d, *model3_6_weight_d, *model3_6_bias_d,
         *model4_0_weight_d, *model4_0_bias_d, *model4_2_weight_d, *model4_2_bias_d, *model4_4_weight_d, *model4_4_bias_d, *model4_6_weight_d, *model4_6_bias_d,
         *model5_0_weight_d, *model5_0_bias_d, *model5_2_weight_d, *model5_2_bias_d, *model5_4_weight_d, *model5_4_bias_d, *model5_6_weight_d, *model5_6_bias_d,
         *model6_0_weight_d, *model6_0_bias_d, *model6_2_weight_d, *model6_2_bias_d, *model6_4_weight_d, *model6_4_bias_d, *model6_6_weight_d, *model6_6_bias_d,
         *model7_0_weight_d, *model7_0_bias_d, *model7_2_weight_d, *model7_2_bias_d, *model7_4_weight_d, *model7_4_bias_d, *model7_6_weight_d, *model7_6_bias_d,
         *model8_0_weight_d, *model8_0_bias_d, *model8_2_weight_d, *model8_2_bias_d, *model8_4_weight_d, *model8_4_bias_d, *model8_6_weight_d, *model8_6_bias_d,
         *model_out_weight_d,
         *model1_4_running_mean_d, *model1_4_running_var_d,
         *model2_4_running_mean_d, *model2_4_running_var_d,
         *model3_6_running_mean_d, *model3_6_running_var_d,
         *model4_6_running_mean_d, *model4_6_running_var_d,
         *model5_6_running_mean_d, *model5_6_running_var_d,
         *model6_6_running_mean_d, *model6_6_running_var_d,
         *model7_6_running_mean_d, *model7_6_running_var_d;
} Weights_cud;
static Weights_cud wsc[NUM_GPU];

static Tensor Make3DTensor(int C, int H, int W);

// Layers
void NormalizeL_cu(Tensor input, Tensor output, float *input_d, float *output_d);
void Conv2d_cu(Tensor input, Tensor weight, Tensor bias, Tensor output, int stride, int pad, int dilation, bool has_bias, float *input_d, float *weight_d, float *bias_d, float *output_d, bool with_relu);
void Conv2d_cu_(Tensor input, Tensor weight, Tensor bias, Tensor output, int stride, int pad, int dilation, float *input_d, float *weight_d, float *bias_d, float *output_d, int last_dim);
void BatchNorm2d_cu(Tensor input, Tensor weight, Tensor bias, Tensor running_mean, Tensor running_var, Tensor output, float *input_d, float *weight_d, float *bias_d, float *running_mean_d, float *running_var_d, float *output_d);
void ConvTranspose2dReLU_cu(Tensor input, Tensor weight, Tensor bias, Tensor output, int stride, int pad, float *input_d, float *weight_d, float *bias_d, float *output_d);
void Softmax_cu(Tensor input, Tensor output, float *input_d, float *output_d);
void UpsampleUnnormalize_cu(Tensor input, Tensor output, float scale_factor, float *input_d, float *output_d);

// Public APIs
void ColorizerInit();
void Colorize(float* input, float* network, float* output, int N);
void ColorizerFinalize();

void Process_CL(float* input, float* network, float* output, int i, Weights ws, int si, int gi);

// MPI
void alloc_heap(float **m, int size);

/*
 * Definitions
 */
void ColorizerInit() {
  /*
   * You can do input-independent jobs here.
   * e.g., Getting OpenCL Platform, allocating feature maps, ...
   * Execution time of this function is not measured, so do as much as possible!
   */
  static int H = 256, W = 256;

  for(int i=0;i<CHK_LEN;i++) {
    fss[i].fm_normalize_l = Make3DTensor(1, 256, 256);
    fss[i].fm1_0 = Make3DTensor(64, 256, 256);
    fss[i].fm1_1 = Make3DTensor(64, 128, 128);
    fss[i].fm1_2 = Make3DTensor(64, 128, 128);
    fss[i].fm2_0 = Make3DTensor(128, 128, 128);
    fss[i].fm2_1 = Make3DTensor(128, 64, 64);
    fss[i].fm2_2 = Make3DTensor(128, 64, 64);
    fss[i].fm3_0 = Make3DTensor(256, 64, 64);
    fss[i].fm3_1 = Make3DTensor(256, 64, 64);
    fss[i].fm3_2 = Make3DTensor(256, 32, 32);
    fss[i].fm3_3 = Make3DTensor(256, 32, 32);
    fss[i].fm4_0 = Make3DTensor(512, 32, 32);
    fss[i].fm4_1 = Make3DTensor(512, 32, 32);
    fss[i].fm4_2 = Make3DTensor(512, 32, 32);
    fss[i].fm4_3 = Make3DTensor(512, 32, 32);
    fss[i].fm5_0 = Make3DTensor(512, 32, 32);
    fss[i].fm5_1 = Make3DTensor(512, 32, 32);
    fss[i].fm5_2 = Make3DTensor(512, 32, 32);
    fss[i].fm5_3 = Make3DTensor(512, 32, 32);
    fss[i].fm6_0 = Make3DTensor(512, 32, 32);
    fss[i].fm6_1 = Make3DTensor(512, 32, 32);
    fss[i].fm6_2 = Make3DTensor(512, 32, 32);
    fss[i].fm6_3 = Make3DTensor(512, 32, 32);
    fss[i].fm7_0 = Make3DTensor(512, 32, 32);
    fss[i].fm7_1 = Make3DTensor(512, 32, 32);
    fss[i].fm7_2 = Make3DTensor(512, 32, 32);
    fss[i].fm7_3 = Make3DTensor(512, 32, 32);
    fss[i].fm8_0 = Make3DTensor(256, 64, 64);
    fss[i].fm8_1 = Make3DTensor(256, 64, 64);
    fss[i].fm8_2 = Make3DTensor(256, 64, 64);
    fss[i].fm8_3 = Make3DTensor(313, 64, 64);
    fss[i].fm_softmax = Make3DTensor(313, 64, 64);
    fss[i].fm_model_out = Make3DTensor(2, 64, 64);
    fss[i].fm_upsample4 = Make3DTensor(2, 256, 256);
  }
  
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

  for(int i=0;i<CHK_LEN;i++) {
    CUDA_CALL( cudaSetDevice(i%num_devices); )
    
    CUDA_CALL( cudaMalloc(&fscs[i].image_L_d, 1 * H * W * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm_normalize_l_d, 1 * 256 * 256 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm1_0_d, 64 * 256 * 256 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm1_1_d, 64 * 128 * 128 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm1_2_d, 64 * 128 * 128 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm2_0_d, 128 * 128 * 128 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm2_1_d, 128 * 64 * 64 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm2_2_d, 128 * 64 * 64 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm3_0_d, 256 * 64 * 64 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm3_1_d, 256 * 64 * 64 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm3_2_d, 256 * 32 * 32 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm3_3_d, 256 * 32 * 32 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm4_0_d, 512 * 32 * 32 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm4_1_d, 512 * 32 * 32 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm4_2_d, 512 * 32 * 32 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm4_3_d, 512 * 32 * 32 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm5_0_d, 512 * 32 * 32 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm5_1_d, 512 * 32 * 32 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm5_2_d, 512 * 32 * 32 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm5_3_d, 512 * 32 * 32 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm6_0_d, 512 * 32 * 32 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm6_1_d, 512 * 32 * 32 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm6_2_d, 512 * 32 * 32 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm6_3_d, 512 * 32 * 32 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm7_0_d, 512 * 32 * 32 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm7_1_d, 512 * 32 * 32 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm7_2_d, 512 * 32 * 32 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm7_3_d, 512 * 32 * 32 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm8_0_d, 256 * 64 * 64 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm8_1_d, 256 * 64 * 64 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm8_2_d, 256 * 64 * 64 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm8_3_d, 313 * 64 * 64 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm_softmax_d, 313 * 64 * 64 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm_model_out_d, 2 * 64 * 64 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].fm_upsample4_d, 2 * 256 * 256 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&fscs[i].image_AB_d, 2 * H * W * sizeof(float)); )
  }

  for(int i=0;i<(int)num_devices;i++) {
    CUDA_CALL( cudaSetDevice(i%num_devices); )
    
    CUDA_CALL( cudaMalloc(&wsc[i].model1_0_weight_d, 64 * 1 * 3 * 3 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model1_0_bias_d, 64 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model1_2_weight_d, 64 * 64 * 3 * 3 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model1_2_bias_d, 64 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model1_4_weight_d, 64 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model1_4_bias_d, 64 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model2_0_weight_d, 128 * 64 * 3 * 3 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model2_0_bias_d, 128 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model2_2_weight_d, 128 * 128 * 3 * 3 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model2_2_bias_d, 128 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model2_4_weight_d, 128 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model2_4_bias_d, 128 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model3_0_weight_d, 256 * 128 * 3 * 3 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model3_0_bias_d, 256 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model3_2_weight_d, 256 * 256 * 3 * 3 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model3_2_bias_d, 256 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model3_4_weight_d, 256 * 256 * 3 * 3 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model3_4_bias_d, 256 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model3_6_weight_d, 256 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model3_6_bias_d, 256 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model4_0_weight_d, 512 * 256 * 3 * 3 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model4_0_bias_d, 512 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model4_2_weight_d, 512 * 512 * 3 * 3 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model4_2_bias_d, 512 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model4_4_weight_d, 512 * 512 * 3 * 3 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model4_4_bias_d, 512 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model4_6_weight_d, 512 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model4_6_bias_d, 512 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model5_0_weight_d, 512 * 512 * 3 * 3 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model5_0_bias_d, 512 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model5_2_weight_d, 512 * 512 * 3 * 3 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model5_2_bias_d, 512 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model5_4_weight_d, 512 * 512 * 3 * 3 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model5_4_bias_d, 512 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model5_6_weight_d, 512 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model5_6_bias_d, 512 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model6_0_weight_d, 512 * 512 * 3 * 3 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model6_0_bias_d, 512 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model6_2_weight_d, 512 * 512 * 3 * 3 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model6_2_bias_d, 512 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model6_4_weight_d, 512 * 512 * 3 * 3 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model6_4_bias_d, 512 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model6_6_weight_d, 512 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model6_6_bias_d, 512 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model7_0_weight_d, 512 * 512 * 3 * 3 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model7_0_bias_d, 512 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model7_2_weight_d, 512 * 512 * 3 * 3 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model7_2_bias_d, 512 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model7_4_weight_d, 512 * 512 * 3 * 3 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model7_4_bias_d, 512 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model7_6_weight_d, 512 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model7_6_bias_d, 512 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model8_0_weight_d, 512 * 256 * 4 * 4 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model8_0_bias_d, 256 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model8_2_weight_d, 256 * 256 * 3 * 3 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model8_2_bias_d, 256 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model8_4_weight_d, 256 * 256 * 3 * 3 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model8_4_bias_d, 256 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model8_6_weight_d, 313 * 256 * 1 * 1 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model8_6_bias_d, 313 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model_out_weight_d, 2 * 313 * 1 * 1 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model1_4_running_mean_d, 64 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model1_4_running_var_d, 64 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model2_4_running_mean_d, 128 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model2_4_running_var_d, 128 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model3_6_running_mean_d, 256 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model3_6_running_var_d, 256 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model4_6_running_mean_d, 512 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model4_6_running_var_d, 512 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model5_6_running_mean_d, 512 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model5_6_running_var_d, 512 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model6_6_running_mean_d, 512 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model6_6_running_var_d, 512 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model7_6_running_mean_d, 512 * sizeof(float)); )
    CUDA_CALL( cudaMalloc(&wsc[i].model7_6_running_var_d, 512 * sizeof(float)); )
  }
}

void Colorize(float* input, float* network, float* output, int N) {
  /*
   * !!!! CAUTION !!!!
   * Like your previous MPI homework, all inputs (input, network, output, and even N)
   * are only given to rank 0 process. You should manually:
   *   1. allocate buffers on rank >0 processes
   *   2. send inputs from rank 0 to others
   *   3. gather outputs from others to rank 0
   */
  _MPI_Bcast(&N);
  bool div_32 = N%32==0;
  if(div_32) {
    if (mpi_rank!=0) {
      alloc_heap(&input, CHK_DIV/mpi_size*1*256*256*sizeof(float));
      alloc_heap(&network, NETW_SIZE*sizeof(float));
      alloc_heap(&output, CHK_DIV/mpi_size*2*256*256*sizeof(float));
    }
    __MPI_Bcast(network);
  }

  // Split network into parameters
  float* offset = network;
  Tensor model1_0_weight{offset, {64, 1, 3, 3}}; offset += 576;
  Tensor model1_0_bias{offset, {64}}; offset += 64;
  Tensor model1_2_weight{offset, {64, 64, 3, 3}}; offset += 36864;
  Tensor model1_2_bias{offset, {64}}; offset += 64;
  Tensor model1_4_weight{offset, {64}}; offset += 64;
  Tensor model1_4_bias{offset, {64}}; offset += 64;
  Tensor model2_0_weight{offset, {128, 64, 3, 3}}; offset += 73728;
  Tensor model2_0_bias{offset, {128}}; offset += 128;
  Tensor model2_2_weight{offset, {128, 128, 3, 3}}; offset += 147456;
  Tensor model2_2_bias{offset, {128}}; offset += 128;
  Tensor model2_4_weight{offset, {128}}; offset += 128;
  Tensor model2_4_bias{offset, {128}}; offset += 128;
  Tensor model3_0_weight{offset, {256, 128, 3, 3}}; offset += 294912;
  Tensor model3_0_bias{offset, {256}}; offset += 256;
  Tensor model3_2_weight{offset, {256, 256, 3, 3}}; offset += 589824;
  Tensor model3_2_bias{offset, {256}}; offset += 256;
  Tensor model3_4_weight{offset, {256, 256, 3, 3}}; offset += 589824;
  Tensor model3_4_bias{offset, {256}}; offset += 256;
  Tensor model3_6_weight{offset, {256}}; offset += 256;
  Tensor model3_6_bias{offset, {256}}; offset += 256;
  Tensor model4_0_weight{offset, {512, 256, 3, 3}}; offset += 1179648;
  Tensor model4_0_bias{offset, {512}}; offset += 512;
  Tensor model4_2_weight{offset, {512, 512, 3, 3}}; offset += 2359296;
  Tensor model4_2_bias{offset, {512}}; offset += 512;
  Tensor model4_4_weight{offset, {512, 512, 3, 3}}; offset += 2359296;
  Tensor model4_4_bias{offset, {512}}; offset += 512;
  Tensor model4_6_weight{offset, {512}}; offset += 512;
  Tensor model4_6_bias{offset, {512}}; offset += 512;
  Tensor model5_0_weight{offset, {512, 512, 3, 3}}; offset += 2359296;
  Tensor model5_0_bias{offset, {512}}; offset += 512;
  Tensor model5_2_weight{offset, {512, 512, 3, 3}}; offset += 2359296;
  Tensor model5_2_bias{offset, {512}}; offset += 512;
  Tensor model5_4_weight{offset, {512, 512, 3, 3}}; offset += 2359296;
  Tensor model5_4_bias{offset, {512}}; offset += 512;
  Tensor model5_6_weight{offset, {512}}; offset += 512;
  Tensor model5_6_bias{offset, {512}}; offset += 512;
  Tensor model6_0_weight{offset, {512, 512, 3, 3}}; offset += 2359296;
  Tensor model6_0_bias{offset, {512}}; offset += 512;
  Tensor model6_2_weight{offset, {512, 512, 3, 3}}; offset += 2359296;
  Tensor model6_2_bias{offset, {512}}; offset += 512;
  Tensor model6_4_weight{offset, {512, 512, 3, 3}}; offset += 2359296;
  Tensor model6_4_bias{offset, {512}}; offset += 512;
  Tensor model6_6_weight{offset, {512}}; offset += 512;
  Tensor model6_6_bias{offset, {512}}; offset += 512;
  Tensor model7_0_weight{offset, {512, 512, 3, 3}}; offset += 2359296;
  Tensor model7_0_bias{offset, {512}}; offset += 512;
  Tensor model7_2_weight{offset, {512, 512, 3, 3}}; offset += 2359296;
  Tensor model7_2_bias{offset, {512}}; offset += 512;
  Tensor model7_4_weight{offset, {512, 512, 3, 3}}; offset += 2359296;
  Tensor model7_4_bias{offset, {512}}; offset += 512;
  Tensor model7_6_weight{offset, {512}}; offset += 512;
  Tensor model7_6_bias{offset, {512}}; offset += 512;
  Tensor model8_0_weight{offset, {512, 256, 4, 4}}; offset += 2097152;
  Tensor model8_0_bias{offset, {256}}; offset += 256;
  Tensor model8_2_weight{offset, {256, 256, 3, 3}}; offset += 589824;
  Tensor model8_2_bias{offset, {256}}; offset += 256;
  Tensor model8_4_weight{offset, {256, 256, 3, 3}}; offset += 589824;
  Tensor model8_4_bias{offset, {256}}; offset += 256;
  Tensor model8_6_weight{offset, {313, 256, 1, 1}}; offset += 80128;
  Tensor model8_6_bias{offset, {313}}; offset += 313;
  Tensor model_out_weight{offset, {2, 313, 1, 1}}; offset += 626;
  Tensor model1_4_running_mean{offset, {64}}; offset += 64;
  Tensor model1_4_running_var{offset, {64}}; offset += 64;
  Tensor model2_4_running_mean{offset, {128}}; offset += 128;
  Tensor model2_4_running_var{offset, {128}}; offset += 128;
  Tensor model3_6_running_mean{offset, {256}}; offset += 256;
  Tensor model3_6_running_var{offset, {256}}; offset += 256;
  Tensor model4_6_running_mean{offset, {512}}; offset += 512;
  Tensor model4_6_running_var{offset, {512}}; offset += 512;
  Tensor model5_6_running_mean{offset, {512}}; offset += 512;
  Tensor model5_6_running_var{offset, {512}}; offset += 512;
  Tensor model6_6_running_mean{offset, {512}}; offset += 512;
  Tensor model6_6_running_var{offset, {512}}; offset += 512;
  Tensor model7_6_running_mean{offset, {512}}; offset += 512;
  Tensor model7_6_running_var{offset, {512}}; offset += 512;

  if(div_32 || mpi_rank == 0) {
    for(int i=0;i<(int)num_devices;i++) {
      CUDA_CALL( cudaSetDevice(i); )
      
      CUDA_CALL( cudaMemcpy(wsc[i].model1_0_weight_d, model1_0_weight.buf, 64 * 1 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model1_0_bias_d, model1_0_bias.buf, 64 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model1_2_weight_d, model1_2_weight.buf, 64 * 64 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);   )
      CUDA_CALL( cudaMemcpy(wsc[i].model1_2_bias_d, model1_2_bias.buf, 64 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model1_4_weight_d, model1_4_weight.buf, 64 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model1_4_bias_d, model1_4_bias.buf, 64 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model2_0_weight_d, model2_0_weight.buf, 128 * 64 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);  )
      CUDA_CALL( cudaMemcpy(wsc[i].model2_0_bias_d, model2_0_bias.buf, 128 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model2_2_weight_d, model2_2_weight.buf, 128 * 128 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model2_2_bias_d, model2_2_bias.buf, 128 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model2_4_weight_d, model2_4_weight.buf, 128 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model2_4_bias_d, model2_4_bias.buf, 128 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model3_0_weight_d, model3_0_weight.buf, 256 * 128 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model3_0_bias_d, model3_0_bias.buf, 256 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model3_2_weight_d, model3_2_weight.buf, 256 * 256 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model3_2_bias_d, model3_2_bias.buf, 256 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model3_4_weight_d, model3_4_weight.buf, 256 * 256 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model3_4_bias_d, model3_4_bias.buf, 256 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model3_6_weight_d, model3_6_weight.buf, 256 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model3_6_bias_d, model3_6_bias.buf, 256 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model4_0_weight_d, model4_0_weight.buf, 512 * 256 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model4_0_bias_d, model4_0_bias.buf, 512 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model4_2_weight_d, model4_2_weight.buf, 512 * 512 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model4_2_bias_d, model4_2_bias.buf, 512 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model4_4_weight_d, model4_4_weight.buf, 512 * 512 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model4_4_bias_d, model4_4_bias.buf, 512 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model4_6_weight_d, model4_6_weight.buf, 512 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model4_6_bias_d, model4_6_bias.buf, 512 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model5_0_weight_d, model5_0_weight.buf, 512 * 512 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model5_0_bias_d, model5_0_bias.buf, 512 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model5_2_weight_d, model5_2_weight.buf, 512 * 512 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model5_2_bias_d, model5_2_bias.buf, 512 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model5_4_weight_d, model5_4_weight.buf, 512 * 512 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model5_4_bias_d, model5_4_bias.buf, 512 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model5_6_weight_d, model5_6_weight.buf, 512 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model5_6_bias_d, model5_6_bias.buf, 512 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model6_0_weight_d, model6_0_weight.buf, 512 * 512 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model6_0_bias_d, model6_0_bias.buf, 512 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model6_2_weight_d, model6_2_weight.buf, 512 * 512 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model6_2_bias_d, model6_2_bias.buf, 512 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model6_4_weight_d, model6_4_weight.buf, 512 * 512 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model6_4_bias_d, model6_4_bias.buf, 512 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model6_6_weight_d, model6_6_weight.buf, 512 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model6_6_bias_d, model6_6_bias.buf, 512 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model7_0_weight_d, model7_0_weight.buf, 512 * 512 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model7_0_bias_d, model7_0_bias.buf, 512 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model7_2_weight_d, model7_2_weight.buf, 512 * 512 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model7_2_bias_d, model7_2_bias.buf, 512 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model7_4_weight_d, model7_4_weight.buf, 512 * 512 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model7_4_bias_d, model7_4_bias.buf, 512 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model7_6_weight_d, model7_6_weight.buf, 512 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model7_6_bias_d, model7_6_bias.buf, 512 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model8_0_weight_d, model8_0_weight.buf, 512 * 256 * 4 * 4 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model8_0_bias_d, model8_0_bias.buf, 256 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model8_2_weight_d, model8_2_weight.buf, 256 * 256 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model8_2_bias_d, model8_2_bias.buf, 256 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model8_4_weight_d, model8_4_weight.buf, 256 * 256 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model8_4_bias_d, model8_4_bias.buf, 256 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model8_6_weight_d, model8_6_weight.buf, 313 * 256 * 1 * 1 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model8_6_bias_d, model8_6_bias.buf, 313 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model_out_weight_d, model_out_weight.buf, 2 * 313 * 1 * 1 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model1_4_running_mean_d, model1_4_running_mean.buf, 64 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model1_4_running_var_d, model1_4_running_var.buf, 64 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model2_4_running_mean_d, model2_4_running_mean.buf, 128 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model2_4_running_var_d, model2_4_running_var.buf, 128 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model3_6_running_mean_d, model3_6_running_mean.buf, 256 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model3_6_running_var_d, model3_6_running_var.buf, 256 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model4_6_running_mean_d, model4_6_running_mean.buf, 512 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model4_6_running_var_d, model4_6_running_var.buf, 512 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model5_6_running_mean_d, model5_6_running_mean.buf, 512 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model5_6_running_var_d, model5_6_running_var.buf, 512 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model6_6_running_mean_d, model6_6_running_mean.buf, 512 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model6_6_running_var_d, model6_6_running_var.buf, 512 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model7_6_running_mean_d, model7_6_running_mean.buf, 512 * sizeof(float), cudaMemcpyHostToDevice); )
      CUDA_CALL( cudaMemcpy(wsc[i].model7_6_running_var_d, model7_6_running_var.buf, 512 * sizeof(float), cudaMemcpyHostToDevice); )

      //cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
      //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    }
  }

  Weights ws{&model1_0_weight, &model1_0_bias, &model1_2_weight, &model1_2_bias, &model1_4_weight, &model1_4_bias, 
            &model2_0_weight, &model2_0_bias, &model2_2_weight, &model2_2_bias, &model2_4_weight, &model2_4_bias, 
            &model3_0_weight, &model3_0_bias, &model3_2_weight, &model3_2_bias, &model3_4_weight, &model3_4_bias, &model3_6_weight, &model3_6_bias,
            &model4_0_weight, &model4_0_bias, &model4_2_weight, &model4_2_bias, &model4_4_weight, &model4_4_bias, &model4_6_weight, &model4_6_bias,
            &model5_0_weight, &model5_0_bias, &model5_2_weight, &model5_2_bias, &model5_4_weight, &model5_4_bias, &model5_6_weight, &model5_6_bias,
            &model6_0_weight, &model6_0_bias, &model6_2_weight, &model6_2_bias, &model6_4_weight, &model6_4_bias, &model6_6_weight, &model6_6_bias,
            &model7_0_weight, &model7_0_bias, &model7_2_weight, &model7_2_bias, &model7_4_weight, &model7_4_bias, &model7_6_weight, &model7_6_bias,
            &model8_0_weight, &model8_0_bias, &model8_2_weight, &model8_2_bias, &model8_4_weight, &model8_4_bias, &model8_6_weight, &model8_6_bias,
            &model_out_weight,
            &model1_4_running_mean, &model1_4_running_var,
            &model2_4_running_mean, &model2_4_running_var,
            &model3_6_running_mean, &model3_6_running_var,
            &model4_6_running_mean, &model4_6_running_var,
            &model5_6_running_mean, &model5_6_running_var,
            &model6_6_running_mean, &model6_6_running_var,
            &model7_6_running_mean, &model7_6_running_var};

  // Let's process i-th image
  const int end_ind = div_32 ? ((mpi_rank==0) ? CHK_LEFT+CHK_DIV/mpi_size : CHK_DIV/mpi_size)
                             : ((mpi_rank==0) ? 32 : 0);
  for(int bi = 0; bi < N/CHK_LEN*CHK_LEN; bi+=CHK_LEN) {
    const int pass_chunk_in = (mpi_rank==0) ? bi*256*256 + CHK_LEFT*256*256 : 0;
    const int pass_chunk_ind = (mpi_rank==0) ? bi : 0;
    const int pass_out = (mpi_rank==0) ? bi*2*256*256 + CHK_LEFT*2*256*256 : 0;

    if (div_32) {
      _MPI_Scatter(input + pass_chunk_in, CHK_DIV/mpi_size*256*256, input + pass_chunk_in, CHK_DIV/mpi_size*256*256, 0); 
    }

#pragma omp parallel for num_threads(CHK_LEN)
    for(int si=0;si<end_ind;si++) {
      Process_CL(input, network, output, pass_chunk_ind+si, ws, si, si%num_devices);
    }

    if (div_32) {
      _MPI_Gather(output + pass_out, CHK_DIV/mpi_size*2*256*256, output + pass_out, CHK_DIV/mpi_size*2*256*256, 0); 
    }
  }
  if(mpi_rank==0) {
    int boundary = N/CHK_LEN*CHK_LEN;
#pragma omp parallel for num_threads(CHK_LEN)
    for(int si = 0; si < N-boundary; ++si) {
      Process_CL(input, network, output, boundary+si, ws, si, si%num_devices);
    }
  }

  if(div_32 && mpi_rank!=0) {
    free(input);
    free(network);
    free(output);
  }
}

void Process_CL(float* input, float* network, float* output, int i, Weights ws, int si, const int gi) {
    static int H = 256, W = 256;

    // Find i-th image in input buffer
    Tensor image_L{input + i * H * W, {1, H, W}};

    // Fine location to write i-th result in output buffer
    Tensor image_AB{output + i * 2 * H * W, {2, H, W}};

    CUDA_CALL( cudaSetDevice(si%4); )

    // NormalizeL
    //timer_reset(1); timer_start(1);
    CUDA_CALL( cudaMemcpy(fscs[si].image_L_d, image_L.buf, 1 * H * W * sizeof(float), cudaMemcpyHostToDevice); )
    NormalizeL_cu(image_L, fss[si].fm_normalize_l, fscs[si].image_L_d, fscs[si].fm_normalize_l_d);
    //PRINTF_WITH_RANK("Normalize done! (%f s)", timer_read(1));

    /*
     * Block 1
     * Comments may help you debug.
     */
    //timer_reset(1); timer_start(1);
    Conv2d_cu(fss[si].fm_normalize_l, *ws.model1_0_weight, *ws.model1_0_bias, fss[si].fm1_0, 1, 1, 1, true, fscs[si].fm_normalize_l_d, wsc[gi].model1_0_weight_d, wsc[gi].model1_0_bias_d, fscs[si].fm1_0_d, true);
    Conv2d_cu_(fss[si].fm1_0, *ws.model1_2_weight, *ws.model1_2_bias, fss[si].fm1_1, 2, 1, 1, fscs[si].fm1_0_d, wsc[gi].model1_2_weight_d, wsc[gi].model1_2_bias_d, fscs[si].fm1_1_d, 64);
    BatchNorm2d_cu(fss[si].fm1_1, *ws.model1_4_weight, *ws.model1_4_bias, *ws.model1_4_running_mean, *ws.model1_4_running_var, fss[si].fm1_2, fscs[si].fm1_1_d, wsc[gi].model1_4_weight_d, wsc[gi].model1_4_bias_d, wsc[gi].model1_4_running_mean_d, wsc[gi].model1_4_running_var_d, fscs[si].fm1_2_d);
    //PRINTF_WITH_RANK("Block 1 done! (%f s)", timer_read(1));

    /*
     * Block 2
     */
    //timer_reset(1); timer_start(1);
    Conv2d_cu_(fss[si].fm1_2, *ws.model2_0_weight, *ws.model2_0_bias, fss[si].fm2_0, 1, 1, 1, fscs[si].fm1_2_d, wsc[gi].model2_0_weight_d, wsc[gi].model2_0_bias_d, fscs[si].fm2_0_d, 64);
    Conv2d_cu_(fss[si].fm2_0, *ws.model2_2_weight, *ws.model2_2_bias, fss[si].fm2_1, 2, 1, 1, fscs[si].fm2_0_d, wsc[gi].model2_2_weight_d, wsc[gi].model2_2_bias_d, fscs[si].fm2_1_d, 64);
    BatchNorm2d_cu(fss[si].fm2_1, *ws.model2_4_weight, *ws.model2_4_bias, *ws.model2_4_running_mean, *ws.model2_4_running_var, fss[si].fm2_2, fscs[si].fm2_1_d, wsc[gi].model2_4_weight_d, wsc[gi].model2_4_bias_d, wsc[gi].model2_4_running_mean_d, wsc[gi].model2_4_running_var_d, fscs[si].fm2_2_d);
    //PRINTF_WITH_RANK("Block 2 done! (%f s)", timer_read(1));

    /*
     * Block 3
     */
    //timer_reset(1); timer_start(1);
    Conv2d_cu_(fss[si].fm2_2, *ws.model3_0_weight, *ws.model3_0_bias, fss[si].fm3_0, 1, 1, 1, fscs[si].fm2_2_d, wsc[gi].model3_0_weight_d, wsc[gi].model3_0_bias_d, fscs[si].fm3_0_d, 64);
    Conv2d_cu_(fss[si].fm3_0, *ws.model3_2_weight, *ws.model3_2_bias, fss[si].fm3_1, 1, 1, 1, fscs[si].fm3_0_d, wsc[gi].model3_2_weight_d, wsc[gi].model3_2_bias_d, fscs[si].fm3_1_d, 64);
    Conv2d_cu_(fss[si].fm3_1, *ws.model3_4_weight, *ws.model3_4_bias, fss[si].fm3_2, 2, 1, 1, fscs[si].fm3_1_d, wsc[gi].model3_4_weight_d, wsc[gi].model3_4_bias_d, fscs[si].fm3_2_d, 32);
    BatchNorm2d_cu(fss[si].fm3_2, *ws.model3_6_weight, *ws.model3_6_bias, *ws.model3_6_running_mean, *ws.model3_6_running_var, fss[si].fm3_3, fscs[si].fm3_2_d, wsc[gi].model3_6_weight_d, wsc[gi].model3_6_bias_d, wsc[gi].model3_6_running_mean_d, wsc[gi].model3_6_running_var_d, fscs[si].fm3_3_d);
    //PRINTF_WITH_RANK("Block 3 done! (%f s)", timer_read(1));

    /*
     * Block 4
     */
    //timer_reset(1); timer_start(1);
    Conv2d_cu_(fss[si].fm3_3, *ws.model4_0_weight, *ws.model4_0_bias, fss[si].fm4_0, 1, 1, 1, fscs[si].fm3_3_d, wsc[gi].model4_0_weight_d, wsc[gi].model4_0_bias_d, fscs[si].fm4_0_d, 32);
    Conv2d_cu_(fss[si].fm4_0, *ws.model4_2_weight, *ws.model4_2_bias, fss[si].fm4_1, 1, 1, 1, fscs[si].fm4_0_d, wsc[gi].model4_2_weight_d, wsc[gi].model4_2_bias_d, fscs[si].fm4_1_d, 32);
    Conv2d_cu_(fss[si].fm4_1, *ws.model4_4_weight, *ws.model4_4_bias, fss[si].fm4_2, 1, 1, 1, fscs[si].fm4_1_d, wsc[gi].model4_4_weight_d, wsc[gi].model4_4_bias_d, fscs[si].fm4_2_d, 32);
    BatchNorm2d_cu(fss[si].fm4_2, *ws.model4_6_weight, *ws.model4_6_bias, *ws.model4_6_running_mean, *ws.model4_6_running_var, fss[si].fm4_3, fscs[si].fm4_2_d, wsc[gi].model4_6_weight_d, wsc[gi].model4_6_bias_d, wsc[gi].model4_6_running_mean_d, wsc[gi].model4_6_running_var_d, fscs[si].fm4_3_d);
    //PRINTF_WITH_RANK("Block 4 done! (%f s)", timer_read(1));

    /*
     * Block 5
     */
    //timer_reset(1); timer_start(1);
    Conv2d_cu_(fss[si].fm4_3, *ws.model5_0_weight, *ws.model5_0_bias, fss[si].fm5_0, 1, 2, 2, fscs[si].fm4_3_d, wsc[gi].model5_0_weight_d, wsc[gi].model5_0_bias_d, fscs[si].fm5_0_d, 32);
    Conv2d_cu_(fss[si].fm5_0, *ws.model5_2_weight, *ws.model5_2_bias, fss[si].fm5_1, 1, 2, 2, fscs[si].fm5_0_d, wsc[gi].model5_2_weight_d, wsc[gi].model5_2_bias_d, fscs[si].fm5_1_d, 32);
    Conv2d_cu_(fss[si].fm5_1, *ws.model5_4_weight, *ws.model5_4_bias, fss[si].fm5_2, 1, 2, 2, fscs[si].fm5_1_d, wsc[gi].model5_4_weight_d, wsc[gi].model5_4_bias_d, fscs[si].fm5_2_d, 32);
    BatchNorm2d_cu(fss[si].fm5_2, *ws.model5_6_weight, *ws.model5_6_bias, *ws.model5_6_running_mean, *ws.model5_6_running_var, fss[si].fm5_3, fscs[si].fm5_2_d, wsc[gi].model5_6_weight_d, wsc[gi].model5_6_bias_d, wsc[gi].model5_6_running_mean_d, wsc[gi].model5_6_running_var_d, fscs[si].fm5_3_d);
    //PRINTF_WITH_RANK("Block 5 done! (%f s)", timer_read(1));

    /*
     * Block 6
     */
    //timer_reset(1); timer_start(1);
    Conv2d_cu_(fss[si].fm5_3, *ws.model6_0_weight, *ws.model6_0_bias, fss[si].fm6_0, 1, 2, 2, fscs[si].fm5_3_d, wsc[gi].model6_0_weight_d, wsc[gi].model6_0_bias_d, fscs[si].fm6_0_d, 32);
    Conv2d_cu_(fss[si].fm6_0, *ws.model6_2_weight, *ws.model6_2_bias, fss[si].fm6_1, 1, 2, 2, fscs[si].fm6_0_d, wsc[gi].model6_2_weight_d, wsc[gi].model6_2_bias_d, fscs[si].fm6_1_d, 32);
    Conv2d_cu_(fss[si].fm6_1, *ws.model6_4_weight, *ws.model6_4_bias, fss[si].fm6_2, 1, 2, 2, fscs[si].fm6_1_d, wsc[gi].model6_4_weight_d, wsc[gi].model6_4_bias_d, fscs[si].fm6_2_d, 32);
    BatchNorm2d_cu(fss[si].fm6_2, *ws.model6_6_weight, *ws.model6_6_bias, *ws.model6_6_running_mean, *ws.model6_6_running_var, fss[si].fm6_3, fscs[si].fm6_2_d, wsc[gi].model6_6_weight_d, wsc[gi].model6_6_bias_d, wsc[gi].model6_6_running_mean_d, wsc[gi].model6_6_running_var_d, fscs[si].fm6_3_d);
    //PRINTF_WITH_RANK("Block 6 done! (%f s)", timer_read(1));

    /*
     * Block 7
     */
    //timer_reset(1); timer_start(1);
    Conv2d_cu_(fss[si].fm6_3, *ws.model7_0_weight, *ws.model7_0_bias, fss[si].fm7_0, 1, 1, 1, fscs[si].fm6_3_d, wsc[gi].model7_0_weight_d, wsc[gi].model7_0_bias_d, fscs[si].fm7_0_d, 32);
    Conv2d_cu_(fss[si].fm7_0, *ws.model7_2_weight, *ws.model7_2_bias, fss[si].fm7_1, 1, 1, 1, fscs[si].fm7_0_d, wsc[gi].model7_2_weight_d, wsc[gi].model7_2_bias_d, fscs[si].fm7_1_d, 32);
    Conv2d_cu_(fss[si].fm7_1, *ws.model7_4_weight, *ws.model7_4_bias, fss[si].fm7_2, 1, 1, 1, fscs[si].fm7_1_d, wsc[gi].model7_4_weight_d, wsc[gi].model7_4_bias_d, fscs[si].fm7_2_d, 32);
    BatchNorm2d_cu(fss[si].fm7_2, *ws.model7_6_weight, *ws.model7_6_bias, *ws.model7_6_running_mean, *ws.model7_6_running_var, fss[si].fm7_3, fscs[si].fm7_2_d, wsc[gi].model7_6_weight_d, wsc[gi].model7_6_bias_d, wsc[gi].model7_6_running_mean_d, wsc[gi].model7_6_running_var_d, fscs[si].fm7_3_d);
    //PRINTF_WITH_RANK("Block 7 done! (%f s)", timer_read(1));

    /*
     * Block 8
     */
    //timer_reset(1); timer_start(1);
    ConvTranspose2dReLU_cu(fss[si].fm7_3, *ws.model8_0_weight, *ws.model8_0_bias, fss[si].fm8_0, 2, 1, fscs[si].fm7_3_d, wsc[gi].model8_0_weight_d, wsc[gi].model8_0_bias_d, fscs[si].fm8_0_d);
    Conv2d_cu_(fss[si].fm8_0, *ws.model8_2_weight, *ws.model8_2_bias, fss[si].fm8_1, 1, 1, 1, fscs[si].fm8_0_d, wsc[gi].model8_2_weight_d, wsc[gi].model8_2_bias_d, fscs[si].fm8_1_d, 64);
    Conv2d_cu_(fss[si].fm8_1, *ws.model8_4_weight, *ws.model8_4_bias, fss[si].fm8_2, 1, 1, 1, fscs[si].fm8_1_d, wsc[gi].model8_4_weight_d, wsc[gi].model8_4_bias_d, fscs[si].fm8_2_d, 64);
    Conv2d_cu(fss[si].fm8_2, *ws.model8_6_weight, *ws.model8_6_bias, fss[si].fm8_3, 1, 0, 1, true, fscs[si].fm8_2_d, wsc[gi].model8_6_weight_d, wsc[gi].model8_6_bias_d, fscs[si].fm8_3_d, false);       
    //PRINTF_WITH_RANK("Block 8 done! (%f s)", timer_read(1));

    /*
     * Wrap-up block
     */
    //timer_reset(1); timer_start(1);
    Softmax_cu(fss[si].fm8_3, fss[si].fm_softmax, fscs[si].fm8_3_d, fscs[si].fm_softmax_d);
    Conv2d_cu(fss[si].fm_softmax, *ws.model_out_weight, {}, fss[si].fm_model_out, 1, 0, 1, false, fscs[si].fm_softmax_d, wsc[gi].model_out_weight_d, NULL, fscs[si].fm_model_out_d, false);
    UpsampleUnnormalize_cu(fss[si].fm_model_out, image_AB, 4, fscs[si].fm_model_out_d, fscs[si].image_AB_d);
    //PRINTF_WITH_RANK("Block output done! (%f s)", timer_read(1));
}

/*
 * Make a new 3D Tensor. Caller is responsible to free its buf.
 */
Tensor Make3DTensor(int C, int H, int W) {
  return Tensor{(float*)malloc(C * H * W * sizeof(float)), {C, H, W}};
}

void NormalizeL_cu(Tensor input, Tensor output, float *input_d, float *output_d) {
  int H = input.shape[1], W = input.shape[2];
  CHECK_ERROR(input.shape[0] == 1 && output.shape[0] == 1 && output.shape[1] == H && output.shape[2] == W, "Size mismatch");

  dim3 blockDim(1, 128, 1);
  dim3 gridDim(H, W/WLEN_N/128, 1);
  NormalizeL<<<gridDim, blockDim>>>(input_d, output_d);
}

void Conv2d_cu(Tensor input, Tensor weight, Tensor bias, Tensor output, int stride, int pad, int dilation, bool has_bias, float *input_d, float *weight_d, float *bias_d, float *output_d, bool with_relu) {
  int C = input.shape[0], H = input.shape[1], W = input.shape[2];
  int K = weight.shape[0], R = weight.shape[2], S = weight.shape[3];
  int OH = output.shape[1], OW = output.shape[2];
  CHECK_ERROR(OH == (H + 2 * pad - dilation * (R - 1) - 1) / stride + 1, "Output height mismatch");
  CHECK_ERROR(OW == (W + 2 * pad - dilation * (S - 1) - 1) / stride + 1, "Output width mismatch");
  CHECK_ERROR(weight.shape[1] == C && (!has_bias || bias.shape[0] == K) && output.shape[0] == K, "Channel size mismatch");

  int has_bias_int = has_bias ? 1 : 0;
  int with_relu_int = with_relu ? 1 : 0;

  const int blen_1 = 2;
  const int blen_2 = 32;
  dim3 blockDim(1, blen_1, blen_2);
  dim3 gridDim(K, OH/blen_1, OW/WLEN/blen_2);
  Conv2d<<<gridDim, blockDim>>>(input_d, weight_d, bias_d, output_d,
                      stride, pad, dilation, has_bias_int,
                      C, H,
                      K, R,
                      OH,
                      with_relu_int);
}

void Conv2d_cu_(Tensor input, Tensor weight, Tensor bias, Tensor output, int stride, int pad, int dilation, float *input_d, float *weight_d, float *bias_d, float *output_d, int last_dim) {
  int C = input.shape[0], H = input.shape[1], W = input.shape[2];
  int K = weight.shape[0], R = weight.shape[2], S = weight.shape[3];
  int OH = output.shape[1], OW = output.shape[2];
  CHECK_ERROR(OH == (H + 2 * pad - dilation * (R - 1) - 1) / stride + 1, "Output height mismatch");
  CHECK_ERROR(OW == (W + 2 * pad - dilation * (S - 1) - 1) / stride + 1, "Output width mismatch");
  CHECK_ERROR(weight.shape[1] == C && bias.shape[0] == K && output.shape[0] == K, "Channel size mismatch");

  int blen_1 = 1, blen_2 = last_dim/WLEN;

  dim3 blockDim(1, blen_1, blen_2);
  dim3 gridDim(K, OH/blen_1, OW/WLEN/blen_2);
  Conv2d64<<<gridDim, blockDim>>>(input_d, weight_d, bias_d, output_d,
                     stride, pad, dilation,
                     C, H,
                     K,
                     OH);
}

void BatchNorm2d_cu(Tensor input, Tensor weight, Tensor bias, Tensor running_mean, Tensor running_var, Tensor output, float *input_d, float *weight_d, float *bias_d, float *running_mean_d, float *running_var_d, float *output_d) {
  int C = input.shape[0], H = input.shape[1], W = input.shape[2];
  CHECK_ERROR(weight.shape[0] == C && bias.shape[0] == C && running_mean.shape[0] == C && running_var.shape[0] == C, "Shape mismatch");
  CHECK_ERROR(output.shape[0] == C && output.shape[1] == H && output.shape[2] == W, "Shape mismatch");

  const int blen_1 = 2;
  const int blen_2 = 32;
  dim3 blockDim(1, blen_1, blen_2);
  dim3 gridDim(C, H/blen_1, W/blen_2);
  BatchNorm2d<<<gridDim, blockDim>>>(input_d, weight_d, bias_d, running_mean_d, running_var_d, output_d,
                     C, H);
}

void ConvTranspose2dReLU_cu(Tensor input, Tensor weight, Tensor bias, Tensor output, int stride, int pad, float *input_d, float *weight_d, float *bias_d, float *output_d) {
  int C = input.shape[0], H = input.shape[1], W = input.shape[2];
  int K = weight.shape[1], R = weight.shape[2], S = weight.shape[3];
  int OH = output.shape[1], OW = output.shape[2];

  CHECK_ERROR(OH == (H - 1) * stride - 2 * pad + R, "Output height mismatch");
  CHECK_ERROR(OW == (W - 1) * stride - 2 * pad + S, "Output width mismatch");
  CHECK_ERROR(weight.shape[0] == C && bias.shape[0] == K && output.shape[0] == K, "Channel size mismatch");

  const int blen_2 = 64;
  dim3 blockDim(1, 1, blen_2);
  dim3 gridDim(K, OH, OW/blen_2);
  ConvTranspose2dReLU<<<gridDim, blockDim>>>(input_d, weight_d, bias_d, output_d);
}

void Softmax_cu(Tensor input, Tensor output, float *input_d, float *output_d) {
  int C = input.shape[0], H = input.shape[1], W = input.shape[2];
  CHECK_ERROR(output.shape[0] == C && output.shape[1] == H && output.shape[2] == W, "shape mismatch");

  const int blen_1 = 32;
  dim3 blockDim(1, blen_1, 1);
  dim3 gridDim(H, W/blen_1, 1);
  Softmax<<<gridDim, blockDim>>>(input_d, output_d);
}

void UpsampleUnnormalize_cu(Tensor input, Tensor output, float scale_factor, float *input_d, float *output_d) {
  int C = input.shape[0], H = input.shape[1], W = input.shape[2];
  int OH = output.shape[1], OW = output.shape[2];
  CHECK_ERROR(output.shape[0] == C && OH == floorf(H * scale_factor) && OW == floorf(W * scale_factor), "shape mismatch");

  const int blen_2 = 64;
  dim3 blockDim(1, 1, blen_2);
  dim3 gridDim(C, OH, OW/blen_2);
  UpsampleUnnormalize<<<gridDim, blockDim>>>(input_d, output_d);
  
  CUDA_CALL( cudaMemcpy(output.buf, output_d, C * OH * OW * sizeof(float), cudaMemcpyDeviceToHost); )
}

void ColorizerFinalize() {
  // Free buffers we allocated in ColorizerInit
  for(int i=0;i<CHK_LEN;i++) {
    free(fss[i].fm_normalize_l.buf);
    free(fss[i].fm1_0.buf);
    free(fss[i].fm1_1.buf);
    free(fss[i].fm1_2.buf);
    free(fss[i].fm2_0.buf);
    free(fss[i].fm2_1.buf);
    free(fss[i].fm2_2.buf);
    free(fss[i].fm3_0.buf);
    free(fss[i].fm3_1.buf);
    free(fss[i].fm3_2.buf);
    free(fss[i].fm3_3.buf);
    free(fss[i].fm4_0.buf);
    free(fss[i].fm4_1.buf);
    free(fss[i].fm4_2.buf);
    free(fss[i].fm4_3.buf);
    free(fss[i].fm5_0.buf);
    free(fss[i].fm5_1.buf);
    free(fss[i].fm5_2.buf);
    free(fss[i].fm5_3.buf);
    free(fss[i].fm6_0.buf);
    free(fss[i].fm6_1.buf);
    free(fss[i].fm6_2.buf);
    free(fss[i].fm6_3.buf);
    free(fss[i].fm7_0.buf);
    free(fss[i].fm7_1.buf);
    free(fss[i].fm7_2.buf);
    free(fss[i].fm7_3.buf);
    free(fss[i].fm8_0.buf);
    free(fss[i].fm8_1.buf);
    free(fss[i].fm8_2.buf);
    free(fss[i].fm8_3.buf);
    free(fss[i].fm_softmax.buf);
    free(fss[i].fm_model_out.buf);
    free(fss[i].fm_upsample4.buf);
  }
}

void alloc_heap(float **m, int size) {
  *m = (float *) malloc(size);
  if (*m == NULL) {
    printf("Failed to allocate memory for matrix.\n");
    _MPI_Finalize();
    exit(0);
  }
}

