#include "colorizer.h"

#include <cmath>
#include <CL/cl.h>

#include "util.h"
#include "timer.h"

#define NUM_TDS 32

#define CHECK_ERROR_CL(err) \
  if (err != CL_SUCCESS) { \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE); \
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
static Features fss[NUM_TDS];

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

// OpenCL
static cl_int err;
static cl_platform_id platform;
static cl_device_id device;
static cl_context context;
static cl_command_queue write_queue, exec_queue;
static cl_event write_input_e[32];
static cl_program program;
typedef struct Kernels {
  cl_kernel NormalizeL_k;
  cl_kernel Conv2d_k[23];
  cl_kernel BatchNorm2d_k[7];
  cl_kernel ConvTranspose2dReLU_k;
  cl_kernel Softmax_k;
  cl_kernel Upsample_k;
  cl_kernel UnnormalizeAB_k;
} Kernels;
static Kernels kns[32]; 

typedef struct Features_d {
  cl_mem image_L_d, image_AB_d,
         fm_normalize_l_d, 
         fm1_0_d, fm1_1_d, fm1_2_d, 
         fm2_0_d, fm2_1_d, fm2_2_d,
         fm3_0_d, fm3_1_d, fm3_2_d, fm3_3_d,
         fm4_0_d, fm4_1_d, fm4_2_d, fm4_3_d,
         fm5_0_d, fm5_1_d, fm5_2_d, fm5_3_d,
         fm6_0_d, fm6_1_d, fm6_2_d, fm6_3_d,
         fm7_0_d, fm7_1_d, fm7_2_d, fm7_3_d,
         fm8_0_d, fm8_1_d, fm8_2_d, fm8_3_d,
         fm_softmax_d, fm_model_out_d, fm_upsample4_d;
} Features_d;
// Array of Feature maps
static Features_d fsds[NUM_TDS];
typedef struct Weights_d {
  cl_mem model1_0_weight_d, model1_0_bias_d, model1_2_weight_d, model1_2_bias_d, model1_4_weight_d, model1_4_bias_d, 
         model2_0_weight_d, model2_0_bias_d, model2_2_weight_d, model2_2_bias_d, model2_4_weight_d, model2_4_bias_d,
         model3_0_weight_d, model3_0_bias_d, model3_2_weight_d, model3_2_bias_d, model3_4_weight_d, model3_4_bias_d, model3_6_weight_d, model3_6_bias_d, 
         model4_0_weight_d, model4_0_bias_d, model4_2_weight_d, model4_2_bias_d, model4_4_weight_d, model4_4_bias_d, model4_6_weight_d, model4_6_bias_d, 
         model5_0_weight_d, model5_0_bias_d, model5_2_weight_d, model5_2_bias_d, model5_4_weight_d, model5_4_bias_d, model5_6_weight_d, model5_6_bias_d, 
         model6_0_weight_d, model6_0_bias_d, model6_2_weight_d, model6_2_bias_d, model6_4_weight_d, model6_4_bias_d, model6_6_weight_d, model6_6_bias_d,
         model7_0_weight_d, model7_0_bias_d, model7_2_weight_d, model7_2_bias_d, model7_4_weight_d, model7_4_bias_d, model7_6_weight_d, model7_6_bias_d,
         model8_0_weight_d, model8_0_bias_d, model8_2_weight_d, model8_2_bias_d, model8_4_weight_d, model8_4_bias_d, model8_6_weight_d, model8_6_bias_d,
         model_out_weight_d,
         model1_4_running_mean_d, model1_4_running_var_d,
         model2_4_running_mean_d, model2_4_running_var_d,
         model3_6_running_mean_d, model3_6_running_var_d,
         model4_6_running_mean_d, model4_6_running_var_d,
         model5_6_running_mean_d, model5_6_running_var_d,
         model6_6_running_mean_d, model6_6_running_var_d,
         model7_6_running_mean_d, model7_6_running_var_d;
} Weights_d;
static Weights_d wsd;

// Layers
static Tensor Make3DTensor(int C, int H, int W);
//static void DumpTensor(const char* filename, Tensor input, int dim);
static void NormalizeL(Tensor input, Tensor output);
static void Conv2d(Tensor input, Tensor weight, Tensor bias, Tensor output, int stride, int pad, int dilation, bool has_bias = true);
static void ReLU(Tensor inout);
static void BatchNorm2d(Tensor input, Tensor weight, Tensor bias, Tensor running_mean, Tensor running_var, Tensor output, const float eps = 1e-5);
static void ConvTranspose2d(Tensor input, Tensor weight, Tensor bias, Tensor output, int stride, int pad);
static void Softmax(Tensor input, Tensor output);
static void Upsample(Tensor input, Tensor output, float scale_factor);
static void UnnormalizeAB(Tensor input, Tensor output);

void NormalizeL_CL(Tensor input, Tensor output, cl_mem input_d, cl_mem output_d, cl_kernel kernel, int si);
void Conv2d_CL(Tensor input, Tensor weight, Tensor bias, Tensor output, int stride, int pad, int dilation, bool has_bias, cl_mem input_d, cl_mem weight_d, cl_mem bias_d, cl_mem output_d, cl_kernel kernel, bool with_relu);
void BatchNorm2d_CL(Tensor input, Tensor weight, Tensor bias, Tensor running_mean, Tensor running_var, Tensor output, const float eps, cl_mem input_d, cl_mem weight_d, cl_mem bias_d, cl_mem running_mean_d, cl_mem running_var_d, cl_mem output_d, cl_kernel kernel);
void ConvTranspose2dReLU_CL(Tensor input, Tensor weight, Tensor bias, Tensor output, int stride, int pad, cl_mem input_d, cl_mem weight_d, cl_mem bias_d, cl_mem output_d, cl_kernel kernel);
void Softmax_CL(Tensor input, Tensor output, cl_mem input_d, cl_mem output_d, cl_kernel kernel);
void Upsample_CL(Tensor input, Tensor output, float scale_factor, cl_mem input_d, cl_mem output_d, cl_kernel kernel);
void UnnormalizeAB_CL(Tensor input, Tensor output, cl_mem input_d, cl_mem output_d, cl_kernel kernel);

// Public APIs
void ColorizerInit();
void Colorize(float* input, float* network, float* output, int N);
void ColorizerFinalize();

void Process(float* input, float* network, float* output, int i, Weights ws, int si);
void Process_CL(float* input, float* network, float* output, int i, Weights ws, int si);

// OpenCL
static void print_platform_info(cl_platform_id platform);
static void print_device_info(cl_device_id device);
static cl_program create_and_build_program_with_source(cl_context context, cl_device_id device, const char *file_name);

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

  for(int i=0;i<NUM_TDS;i++) {
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

  err = clGetPlatformIDs(1, &platform, NULL);
  CHECK_ERROR_CL(err);
  print_platform_info(platform);

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  CHECK_ERROR_CL(err);
  print_device_info(device);

  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK_ERROR_CL(err);

  write_queue = clCreateCommandQueue(context, device, 0, &err);
  CHECK_ERROR_CL(err);
  exec_queue = clCreateCommandQueue(context, device, 0, &err);
  CHECK_ERROR_CL(err);

  program = create_and_build_program_with_source(context, device, "kernel.cl");

  for(int i=0;i<32;i++) {
    kns[i].NormalizeL_k = clCreateKernel(program, "NormalizeL", &err);
    CHECK_ERROR_CL(err);
    for(int j=0;j<23;j++) {
      kns[i].Conv2d_k[j] = clCreateKernel(program, "Conv2d", &err);
      CHECK_ERROR_CL(err);
    }
    for(int j=0;j<7;j++) {
      kns[i].BatchNorm2d_k[j] = clCreateKernel(program, "BatchNorm2d", &err);
      CHECK_ERROR_CL(err);
    }
    kns[i].ConvTranspose2dReLU_k = clCreateKernel(program, "ConvTranspose2dReLU", &err);
    CHECK_ERROR_CL(err);
    kns[i].Softmax_k = clCreateKernel(program, "Softmax", &err);
    CHECK_ERROR_CL(err);
    kns[i].Upsample_k = clCreateKernel(program, "Upsample", &err);
    CHECK_ERROR_CL(err);
    kns[i].UnnormalizeAB_k = clCreateKernel(program, "UnnormalizeAB", &err);
    CHECK_ERROR_CL(err);
  }

  for(int i=0;i<NUM_TDS;i++) {
    fsds[i].image_L_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 1 * H * W * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm_normalize_l_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 1 * 256 * 256 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm1_0_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 256 * 256 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm1_1_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 128 * 128 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm1_2_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 128 * 128 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm2_0_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 128 * 128 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm2_1_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 64 * 64 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm2_2_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 64 * 64 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm3_0_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 64 * 64 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm3_1_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 64 * 64 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm3_2_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 32 * 32 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm3_3_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 32 * 32 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm4_0_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 32 * 32 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm4_1_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 32 * 32 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm4_2_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 32 * 32 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm4_3_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 32 * 32 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm5_0_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 32 * 32 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm5_1_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 32 * 32 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm5_2_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 32 * 32 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm5_3_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 32 * 32 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm6_0_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 32 * 32 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm6_1_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 32 * 32 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm6_2_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 32 * 32 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm6_3_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 32 * 32 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm7_0_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 32 * 32 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm7_1_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 32 * 32 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm7_2_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 32 * 32 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm7_3_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 32 * 32 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm8_0_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 64 * 64 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm8_1_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 64 * 64 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm8_2_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 64 * 64 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm8_3_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 313 * 64 * 64 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm_softmax_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 313 * 64 * 64 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm_model_out_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * 64 * 64 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].fm_upsample4_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * 256 * 256 * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
    fsds[i].image_AB_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * H * W * sizeof(float), NULL, &err);
    CHECK_ERROR_CL(err);
  }
  wsd.model1_0_weight_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 1 * 3 * 3 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model1_0_bias_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model1_2_weight_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 64 * 3 * 3 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model1_2_bias_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model1_4_weight_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model1_4_bias_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model2_0_weight_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 64 * 3 * 3 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model2_0_bias_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model2_2_weight_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 128 * 3 * 3 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model2_2_bias_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model2_4_weight_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model2_4_bias_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model3_0_weight_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 128 * 3 * 3 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model3_0_bias_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model3_2_weight_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 256 * 3 * 3 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model3_2_bias_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model3_4_weight_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 256 * 3 * 3 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model3_4_bias_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model3_6_weight_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model3_6_bias_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model4_0_weight_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 256 * 3 * 3 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model4_0_bias_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model4_2_weight_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * 3 * 3 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model4_2_bias_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model4_4_weight_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * 3 * 3 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model4_4_bias_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model4_6_weight_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model4_6_bias_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model5_0_weight_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * 3 * 3 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model5_0_bias_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model5_2_weight_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * 3 * 3 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model5_2_bias_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model5_4_weight_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * 3 * 3 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model5_4_bias_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model5_6_weight_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model5_6_bias_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model6_0_weight_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * 3 * 3 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model6_0_bias_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model6_2_weight_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * 3 * 3 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model6_2_bias_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model6_4_weight_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * 3 * 3 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model6_4_bias_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model6_6_weight_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model6_6_bias_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model7_0_weight_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * 3 * 3 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model7_0_bias_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model7_2_weight_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * 3 * 3 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model7_2_bias_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model7_4_weight_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * 3 * 3 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model7_4_bias_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model7_6_weight_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model7_6_bias_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model8_0_weight_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 256 * 4 * 4 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model8_0_bias_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model8_2_weight_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 256 * 3 * 3 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model8_2_bias_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model8_4_weight_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 256 * 3 * 3 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model8_4_bias_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model8_6_weight_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 313 * 256 * 1 * 1 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model8_6_bias_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 313 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model_out_weight_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * 313 * 1 * 1 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model1_4_running_mean_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model1_4_running_var_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model2_4_running_mean_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model2_4_running_var_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model3_6_running_mean_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model3_6_running_var_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model4_6_running_mean_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model4_6_running_var_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model5_6_running_mean_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model5_6_running_var_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model6_6_running_mean_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model6_6_running_var_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model7_6_running_mean_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
  wsd.model7_6_running_var_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, &err);
  CHECK_ERROR_CL(err);
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

  err = clEnqueueWriteBuffer(write_queue, wsd.model1_0_weight_d, CL_FALSE, 0, 64 * 1 * 3 * 3 * sizeof(float), model1_0_weight.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model1_0_bias_d, CL_FALSE, 0, 64 * sizeof(float), model1_0_bias.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model1_2_weight_d, CL_FALSE, 0, 64 * 64 * 3 * 3 * sizeof(float), model1_2_weight.buf, 0, NULL, NULL);  
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model1_2_bias_d, CL_FALSE, 0, 64 * sizeof(float), model1_2_bias.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model1_4_weight_d, CL_FALSE, 0, 64 * sizeof(float), model1_4_weight.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model1_4_bias_d, CL_FALSE, 0, 64 * sizeof(float), model1_4_bias.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model2_0_weight_d, CL_FALSE, 0, 128 * 64 * 3 * 3 * sizeof(float), model2_0_weight.buf, 0, NULL, NULL); 
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model2_0_bias_d, CL_FALSE, 0, 128 * sizeof(float), model2_0_bias.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model2_2_weight_d, CL_FALSE, 0, 128 * 128 * 3 * 3 * sizeof(float), model2_2_weight.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model2_2_bias_d, CL_FALSE, 0, 128 * sizeof(float), model2_2_bias.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model2_4_weight_d, CL_FALSE, 0, 128 * sizeof(float), model2_4_weight.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model2_4_bias_d, CL_FALSE, 0, 128 * sizeof(float), model2_4_bias.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model3_0_weight_d, CL_FALSE, 0, 256 * 128 * 3 * 3 * sizeof(float), model3_0_weight.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model3_0_bias_d, CL_FALSE, 0, 256 * sizeof(float), model3_0_bias.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model3_2_weight_d, CL_FALSE, 0, 256 * 256 * 3 * 3 * sizeof(float), model3_2_weight.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model3_2_bias_d, CL_FALSE, 0, 256 * sizeof(float), model3_2_bias.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model3_4_weight_d, CL_FALSE, 0, 256 * 256 * 3 * 3 * sizeof(float), model3_4_weight.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model3_4_bias_d, CL_FALSE, 0, 256 * sizeof(float), model3_4_bias.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model3_6_weight_d, CL_FALSE, 0, 256 * sizeof(float), model3_6_weight.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model3_6_bias_d, CL_FALSE, 0, 256 * sizeof(float), model3_6_bias.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model4_0_weight_d, CL_FALSE, 0, 512 * 256 * 3 * 3 * sizeof(float), model4_0_weight.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model4_0_bias_d, CL_FALSE, 0, 512 * sizeof(float), model4_0_bias.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model4_2_weight_d, CL_FALSE, 0, 512 * 512 * 3 * 3 * sizeof(float), model4_2_weight.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model4_2_bias_d, CL_FALSE, 0, 512 * sizeof(float), model4_2_bias.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model4_4_weight_d, CL_FALSE, 0, 512 * 512 * 3 * 3 * sizeof(float), model4_4_weight.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model4_4_bias_d, CL_FALSE, 0, 512 * sizeof(float), model4_4_bias.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model4_6_weight_d, CL_FALSE, 0, 512 * sizeof(float), model4_6_weight.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model4_6_bias_d, CL_FALSE, 0, 512 * sizeof(float), model4_6_bias.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model5_0_weight_d, CL_FALSE, 0, 512 * 512 * 3 * 3 * sizeof(float), model5_0_weight.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model5_0_bias_d, CL_FALSE, 0, 512 * sizeof(float), model5_0_bias.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model5_2_weight_d, CL_FALSE, 0, 512 * 512 * 3 * 3 * sizeof(float), model5_2_weight.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model5_2_bias_d, CL_FALSE, 0, 512 * sizeof(float), model5_2_bias.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model5_4_weight_d, CL_FALSE, 0, 512 * 512 * 3 * 3 * sizeof(float), model5_4_weight.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model5_4_bias_d, CL_FALSE, 0, 512 * sizeof(float), model5_4_bias.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model5_6_weight_d, CL_FALSE, 0, 512 * sizeof(float), model5_6_weight.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model5_6_bias_d, CL_FALSE, 0, 512 * sizeof(float), model5_6_bias.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model6_0_weight_d, CL_FALSE, 0, 512 * 512 * 3 * 3 * sizeof(float), model6_0_weight.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model6_0_bias_d, CL_FALSE, 0, 512 * sizeof(float), model6_0_bias.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model6_2_weight_d, CL_FALSE, 0, 512 * 512 * 3 * 3 * sizeof(float), model6_2_weight.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model6_2_bias_d, CL_FALSE, 0, 512 * sizeof(float), model6_2_bias.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model6_4_weight_d, CL_FALSE, 0, 512 * 512 * 3 * 3 * sizeof(float), model6_4_weight.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model6_4_bias_d, CL_FALSE, 0, 512 * sizeof(float), model6_4_bias.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model6_6_weight_d, CL_FALSE, 0, 512 * sizeof(float), model6_6_weight.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model6_6_bias_d, CL_FALSE, 0, 512 * sizeof(float), model6_6_bias.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model7_0_weight_d, CL_FALSE, 0, 512 * 512 * 3 * 3 * sizeof(float), model7_0_weight.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model7_0_bias_d, CL_FALSE, 0, 512 * sizeof(float), model7_0_bias.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model7_2_weight_d, CL_FALSE, 0, 512 * 512 * 3 * 3 * sizeof(float), model7_2_weight.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model7_2_bias_d, CL_FALSE, 0, 512 * sizeof(float), model7_2_bias.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model7_4_weight_d, CL_FALSE, 0, 512 * 512 * 3 * 3 * sizeof(float), model7_4_weight.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model7_4_bias_d, CL_FALSE, 0, 512 * sizeof(float), model7_4_bias.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model7_6_weight_d, CL_FALSE, 0, 512 * sizeof(float), model7_6_weight.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model7_6_bias_d, CL_FALSE, 0, 512 * sizeof(float), model7_6_bias.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model8_0_weight_d, CL_FALSE, 0, 512 * 256 * 4 * 4 * sizeof(float), model8_0_weight.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model8_0_bias_d, CL_FALSE, 0, 256 * sizeof(float), model8_0_bias.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model8_2_weight_d, CL_FALSE, 0, 256 * 256 * 3 * 3 * sizeof(float), model8_2_weight.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model8_2_bias_d, CL_FALSE, 0, 256 * sizeof(float), model8_2_bias.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model8_4_weight_d, CL_FALSE, 0, 256 * 256 * 3 * 3 * sizeof(float), model8_4_weight.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model8_4_bias_d, CL_FALSE, 0, 256 * sizeof(float), model8_4_bias.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model8_6_weight_d, CL_FALSE, 0, 313 * 256 * 1 * 1 * sizeof(float), model8_6_weight.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model8_6_bias_d, CL_FALSE, 0, 313 * sizeof(float), model8_6_bias.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model_out_weight_d, CL_FALSE, 0, 2 * 313 * 1 * 1 * sizeof(float), model_out_weight.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model1_4_running_mean_d, CL_FALSE, 0, 64 * sizeof(float), model1_4_running_mean.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model1_4_running_var_d, CL_FALSE, 0, 64 * sizeof(float), model1_4_running_var.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model2_4_running_mean_d, CL_FALSE, 0, 128 * sizeof(float), model2_4_running_mean.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model2_4_running_var_d, CL_FALSE, 0, 128 * sizeof(float), model2_4_running_var.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model3_6_running_mean_d, CL_FALSE, 0, 256 * sizeof(float), model3_6_running_mean.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model3_6_running_var_d, CL_FALSE, 0, 256 * sizeof(float), model3_6_running_var.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model4_6_running_mean_d, CL_FALSE, 0, 512 * sizeof(float), model4_6_running_mean.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model4_6_running_var_d, CL_FALSE, 0, 512 * sizeof(float), model4_6_running_var.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model5_6_running_mean_d, CL_FALSE, 0, 512 * sizeof(float), model5_6_running_mean.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model5_6_running_var_d, CL_FALSE, 0, 512 * sizeof(float), model5_6_running_var.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model6_6_running_mean_d, CL_FALSE, 0, 512 * sizeof(float), model6_6_running_mean.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model6_6_running_var_d, CL_FALSE, 0, 512 * sizeof(float), model6_6_running_var.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model7_6_running_mean_d, CL_FALSE, 0, 512 * sizeof(float), model7_6_running_mean.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
  err = clEnqueueWriteBuffer(write_queue, wsd.model7_6_running_var_d, CL_FALSE, 0, 512 * sizeof(float), model7_6_running_var.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);

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
  for(int bi = 0; bi < N/NUM_TDS*NUM_TDS; bi+=NUM_TDS) {
#pragma omp parallel for num_threads(NUM_TDS)
    for(int si=0;si<NUM_TDS;si++) {
      Process_CL(input, network, output, bi+si, ws, si);
    }
  }
  int boundary = N/NUM_TDS*NUM_TDS;
#pragma omp parallel for num_threads(NUM_TDS)
  for(int si = 0; si < N-boundary; ++si) {
    Process_CL(input, network, output, boundary+si, ws, si);
  }
}

void Process(float* input, float* network, float* output, int i, Weights ws, int si) {
    static int H = 256, W = 256;

      // Find i-th image in input buffer
    Tensor image_L{input + i * H * W, {1, H, W}};

    // Fine location to write i-th result in output buffer
    Tensor image_AB{output + i * 2 * H * W, {2, H, W}};

    // NormalizeL
    //timer_reset(1); timer_start(1);
    NormalizeL(image_L, fss[si].fm_normalize_l);
    //PRINTF_WITH_RANK("Normalize done! (%f s)", timer_read(1));

    /*
     * Block 1
     * Comments may help you debug.
     */

    //timer_reset(1); timer_start(1);
    Conv2d(fss[si].fm_normalize_l, *ws.model1_0_weight, *ws.model1_0_bias, fss[si].fm1_0, 1, 1, 1);
    ReLU(fss[si].fm1_0);
    Conv2d(fss[si].fm1_0, *ws.model1_2_weight, *ws.model1_2_bias, fss[si].fm1_1, 2, 1, 1);
    ReLU(fss[si].fm1_1);
    BatchNorm2d(fss[si].fm1_1, *ws.model1_4_weight, *ws.model1_4_bias, *ws.model1_4_running_mean, *ws.model1_4_running_var, fss[si].fm1_2);
    //PRINTF_WITH_RANK("Block 1 done! (%f s)", timer_read(1));
    //DumpTensor("fm1_2.txt", fss[si].fm1_2, 3);

    /*
     * Block 2
     */

    //timer_reset(1); timer_start(1);
    Conv2d(fss[si].fm1_2, *ws.model2_0_weight, *ws.model2_0_bias, fss[si].fm2_0, 1, 1, 1);
    ReLU(fss[si].fm2_0);
    Conv2d(fss[si].fm2_0, *ws.model2_2_weight, *ws.model2_2_bias, fss[si].fm2_1, 2, 1, 1);
    ReLU(fss[si].fm2_1);
    BatchNorm2d(fss[si].fm2_1, *ws.model2_4_weight, *ws.model2_4_bias, *ws.model2_4_running_mean, *ws.model2_4_running_var, fss[si].fm2_2);
    //PRINTF_WITH_RANK("Block 2 done! (%f s)", timer_read(1));
    //DumpTensor("fm2_2.txt", fss[si].fm2_2, 3);

    /*
     * Block 3
     */

    //timer_reset(1); timer_start(1);
    Conv2d(fss[si].fm2_2, *ws.model3_0_weight, *ws.model3_0_bias, fss[si].fm3_0, 1, 1, 1);
    ReLU(fss[si].fm3_0);
    Conv2d(fss[si].fm3_0, *ws.model3_2_weight, *ws.model3_2_bias, fss[si].fm3_1, 1, 1, 1);
    ReLU(fss[si].fm3_1);
    Conv2d(fss[si].fm3_1, *ws.model3_4_weight, *ws.model3_4_bias, fss[si].fm3_2, 2, 1, 1);
    ReLU(fss[si].fm3_2);
    BatchNorm2d(fss[si].fm3_2, *ws.model3_6_weight, *ws.model3_6_bias, *ws.model3_6_running_mean, *ws.model3_6_running_var, fss[si].fm3_3);
    //PRINTF_WITH_RANK("Block 3 done! (%f s)", timer_read(1));
    //DumpTensor("fm3_3.txt", fss[si].fm3_3, 3);

    /*
     * Block 4
     */

    //timer_reset(1); timer_start(1);
    Conv2d(fss[si].fm3_3, *ws.model4_0_weight, *ws.model4_0_bias, fss[si].fm4_0, 1, 1, 1);
    ReLU(fss[si].fm4_0);
    Conv2d(fss[si].fm4_0, *ws.model4_2_weight, *ws.model4_2_bias, fss[si].fm4_1, 1, 1, 1);
    ReLU(fss[si].fm4_1);
    Conv2d(fss[si].fm4_1, *ws.model4_4_weight, *ws.model4_4_bias, fss[si].fm4_2, 1, 1, 1);
    ReLU(fss[si].fm4_2);
    BatchNorm2d(fss[si].fm4_2, *ws.model4_6_weight, *ws.model4_6_bias, *ws.model4_6_running_mean, *ws.model4_6_running_var, fss[si].fm4_3);
    //PRINTF_WITH_RANK("Block 4 done! (%f s)", timer_read(1));
    //DumpTensor("fm4_3.txt", fss[si].fm4_3, 3);

    /*
     * Block 5
     */

    //timer_reset(1); timer_start(1);
    Conv2d(fss[si].fm4_3, *ws.model5_0_weight, *ws.model5_0_bias, fss[si].fm5_0, 1, 2, 2);
    ReLU(fss[si].fm5_0);
    Conv2d(fss[si].fm5_0, *ws.model5_2_weight, *ws.model5_2_bias, fss[si].fm5_1, 1, 2, 2);
    ReLU(fss[si].fm5_1);
    Conv2d(fss[si].fm5_1, *ws.model5_4_weight, *ws.model5_4_bias, fss[si].fm5_2, 1, 2, 2);
    ReLU(fss[si].fm5_2);
    BatchNorm2d(fss[si].fm5_2, *ws.model5_6_weight, *ws.model5_6_bias, *ws.model5_6_running_mean, *ws.model5_6_running_var, fss[si].fm5_3);
    //PRINTF_WITH_RANK("Block 5 done! (%f s)", timer_read(1));
    //DumpTensor("fm5_3.txt", fss[si].fm5_3, 3);

    /*
     * Block 6
     */
    //timer_reset(1); timer_start(1);
    Conv2d(fss[si].fm5_3, *ws.model6_0_weight, *ws.model6_0_bias, fss[si].fm6_0, 1, 2, 2);
    ReLU(fss[si].fm6_0);
    Conv2d(fss[si].fm6_0, *ws.model6_2_weight, *ws.model6_2_bias, fss[si].fm6_1, 1, 2, 2);
    ReLU(fss[si].fm6_1);
    Conv2d(fss[si].fm6_1, *ws.model6_4_weight, *ws.model6_4_bias, fss[si].fm6_2, 1, 2, 2);
    ReLU(fss[si].fm6_2);
    BatchNorm2d(fss[si].fm6_2, *ws.model6_6_weight, *ws.model6_6_bias, *ws.model6_6_running_mean, *ws.model6_6_running_var, fss[si].fm6_3);
    //PRINTF_WITH_RANK("Block 6 done! (%f s)", timer_read(1));
    //DumpTensor("fm6_3.txt", fss[si].fm6_3, 3);

    /*
     * Block 7
     */
    //timer_reset(1); timer_start(1);
    Conv2d(fss[si].fm6_3, *ws.model7_0_weight, *ws.model7_0_bias, fss[si].fm7_0, 1, 1, 1);
    ReLU(fss[si].fm7_0);
    Conv2d(fss[si].fm7_0, *ws.model7_2_weight, *ws.model7_2_bias, fss[si].fm7_1, 1, 1, 1);
    ReLU(fss[si].fm7_1);
    Conv2d(fss[si].fm7_1, *ws.model7_4_weight, *ws.model7_4_bias, fss[si].fm7_2, 1, 1, 1);
    ReLU(fss[si].fm7_2);
    BatchNorm2d(fss[si].fm7_2, *ws.model7_6_weight, *ws.model7_6_bias, *ws.model7_6_running_mean, *ws.model7_6_running_var, fss[si].fm7_3);
    //PRINTF_WITH_RANK("Block 7 done! (%f s)", timer_read(1));
    //DumpTensor("fm7_3.txt", fss[si].fm7_3, 3);

    /*
     * Block 8
     */
    //timer_reset(1); timer_start(1);
    ConvTranspose2d(fss[si].fm7_3, *ws.model8_0_weight, *ws.model8_0_bias, fss[si].fm8_0, 2, 1);
    ReLU(fss[si].fm8_0);
    Conv2d(fss[si].fm8_0, *ws.model8_2_weight, *ws.model8_2_bias, fss[si].fm8_1, 1, 1, 1);
    ReLU(fss[si].fm8_1);
    Conv2d(fss[si].fm8_1, *ws.model8_4_weight, *ws.model8_4_bias, fss[si].fm8_2, 1, 1, 1);
    ReLU(fss[si].fm8_2);
    Conv2d(fss[si].fm8_2, *ws.model8_6_weight, *ws.model8_6_bias, fss[si].fm8_3, 1, 0, 1);
    //PRINTF_WITH_RANK("Block 8 done! (%f s)", timer_read(1));
    //DumpTensor("fm8_3.txt", fss[si].fm8_3, 3);

    /*
     * Wrap-up block
     */
    //timer_reset(1); timer_start(1);
    Softmax(fss[si].fm8_3, fss[si].fm_softmax);
    Conv2d(fss[si].fm_softmax, *ws.model_out_weight, {}, fss[si].fm_model_out, 1, 0, 1, false);
    Upsample(fss[si].fm_model_out, fss[si].fm_upsample4, 4);
    UnnormalizeAB(fss[si].fm_upsample4, image_AB);
    //PRINTF_WITH_RANK("Block output done! (%f s)", timer_read(1));
    //DumpTensor("image_AB.txt", image_AB, 3);
}

void Process_CL(float* input, float* network, float* output, int i, Weights ws, int si) {
    static int H = 256, W = 256;

      // Find i-th image in input buffer
    Tensor image_L{input + i * H * W, {1, H, W}};

    // Fine location to write i-th result in output buffer
    Tensor image_AB{output + i * 2 * H * W, {2, H, W}};

    // NormalizeL
    //timer_reset(1); timer_start(1);
    err = clEnqueueWriteBuffer(write_queue, fsds[si].image_L_d, CL_FALSE, 0, 1 * H * W * sizeof(float), image_L.buf, 0, NULL, &write_input_e[si]);
    CHECK_ERROR_CL(err);
    NormalizeL_CL(image_L, fss[si].fm_normalize_l, fsds[si].image_L_d, fsds[si].fm_normalize_l_d, kns[si].NormalizeL_k, si);
    //PRINTF_WITH_RANK("Normalize done! (%f s)", timer_read(1));

    /*
     * Block 1
     * Comments may help you debug.
     */
    //timer_reset(1); timer_start(1);
    Conv2d_CL(fss[si].fm_normalize_l, *ws.model1_0_weight, *ws.model1_0_bias, fss[si].fm1_0, 1, 1, 1, true, fsds[si].fm_normalize_l_d, wsd.model1_0_weight_d, wsd.model1_0_bias_d, fsds[si].fm1_0_d, kns[si].Conv2d_k[0], true);
    Conv2d_CL(fss[si].fm1_0, *ws.model1_2_weight, *ws.model1_2_bias, fss[si].fm1_1, 2, 1, 1, true, fsds[si].fm1_0_d, wsd.model1_2_weight_d, wsd.model1_2_bias_d, fsds[si].fm1_1_d, kns[si].Conv2d_k[1], true);
    BatchNorm2d_CL(fss[si].fm1_1, *ws.model1_4_weight, *ws.model1_4_bias, *ws.model1_4_running_mean, *ws.model1_4_running_var, fss[si].fm1_2, 1e-5, fsds[si].fm1_1_d, wsd.model1_4_weight_d, wsd.model1_4_bias_d, wsd.model1_4_running_mean_d, wsd.model1_4_running_var_d, fsds[si].fm1_2_d, kns[si].BatchNorm2d_k[0]);
    //PRINTF_WITH_RANK("Block 1 done! (%f s)", timer_read(1));
    //DumpTensor("fm1_2.txt", fss[si].fm1_2, 3);

    /*
     * Block 2
     */
    //timer_reset(1); timer_start(1);
    Conv2d_CL(fss[si].fm1_2, *ws.model2_0_weight, *ws.model2_0_bias, fss[si].fm2_0, 1, 1, 1, true, fsds[si].fm1_2_d, wsd.model2_0_weight_d, wsd.model2_0_bias_d, fsds[si].fm2_0_d, kns[si].Conv2d_k[2], true);
    Conv2d_CL(fss[si].fm2_0, *ws.model2_2_weight, *ws.model2_2_bias, fss[si].fm2_1, 2, 1, 1, true, fsds[si].fm2_0_d, wsd.model2_2_weight_d, wsd.model2_2_bias_d, fsds[si].fm2_1_d, kns[si].Conv2d_k[3], true);
    BatchNorm2d_CL(fss[si].fm2_1, *ws.model2_4_weight, *ws.model2_4_bias, *ws.model2_4_running_mean, *ws.model2_4_running_var, fss[si].fm2_2, 1e-5, fsds[si].fm2_1_d, wsd.model2_4_weight_d, wsd.model2_4_bias_d, wsd.model2_4_running_mean_d, wsd.model2_4_running_var_d, fsds[si].fm2_2_d, kns[si].BatchNorm2d_k[1]);
    //PRINTF_WITH_RANK("Block 2 done! (%f s)", timer_read(1));
    //DumpTensor("fm2_2.txt", fss[si].fm2_2, 3);

    /*
     * Block 3
     */
    //timer_reset(1); timer_start(1);
    Conv2d_CL(fss[si].fm2_2, *ws.model3_0_weight, *ws.model3_0_bias, fss[si].fm3_0, 1, 1, 1, true, fsds[si].fm2_2_d, wsd.model3_0_weight_d, wsd.model3_0_bias_d, fsds[si].fm3_0_d, kns[si].Conv2d_k[4], true);
    Conv2d_CL(fss[si].fm3_0, *ws.model3_2_weight, *ws.model3_2_bias, fss[si].fm3_1, 1, 1, 1, true, fsds[si].fm3_0_d, wsd.model3_2_weight_d, wsd.model3_2_bias_d, fsds[si].fm3_1_d, kns[si].Conv2d_k[5], true);
    Conv2d_CL(fss[si].fm3_1, *ws.model3_4_weight, *ws.model3_4_bias, fss[si].fm3_2, 2, 1, 1, true, fsds[si].fm3_1_d, wsd.model3_4_weight_d, wsd.model3_4_bias_d, fsds[si].fm3_2_d, kns[si].Conv2d_k[6], true);
    BatchNorm2d_CL(fss[si].fm3_2, *ws.model3_6_weight, *ws.model3_6_bias, *ws.model3_6_running_mean, *ws.model3_6_running_var, fss[si].fm3_3, 1e-5, fsds[si].fm3_2_d, wsd.model3_6_weight_d, wsd.model3_6_bias_d, wsd.model3_6_running_mean_d, wsd.model3_6_running_var_d, fsds[si].fm3_3_d, kns[si].BatchNorm2d_k[2]);
    //PRINTF_WITH_RANK("Block 3 done! (%f s)", timer_read(1));
    //DumpTensor("fm3_3.txt", fss[si].fm3_3, 3);

    /*
     * Block 4
     */
    //timer_reset(1); timer_start(1);
    Conv2d_CL(fss[si].fm3_3, *ws.model4_0_weight, *ws.model4_0_bias, fss[si].fm4_0, 1, 1, 1, true, fsds[si].fm3_3_d, wsd.model4_0_weight_d, wsd.model4_0_bias_d, fsds[si].fm4_0_d, kns[si].Conv2d_k[7], true);
    Conv2d_CL(fss[si].fm4_0, *ws.model4_2_weight, *ws.model4_2_bias, fss[si].fm4_1, 1, 1, 1, true, fsds[si].fm4_0_d, wsd.model4_2_weight_d, wsd.model4_2_bias_d, fsds[si].fm4_1_d, kns[si].Conv2d_k[8], true);
    Conv2d_CL(fss[si].fm4_1, *ws.model4_4_weight, *ws.model4_4_bias, fss[si].fm4_2, 1, 1, 1, true, fsds[si].fm4_1_d, wsd.model4_4_weight_d, wsd.model4_4_bias_d, fsds[si].fm4_2_d, kns[si].Conv2d_k[9], true);
    BatchNorm2d_CL(fss[si].fm4_2, *ws.model4_6_weight, *ws.model4_6_bias, *ws.model4_6_running_mean, *ws.model4_6_running_var, fss[si].fm4_3, 1e-5, fsds[si].fm4_2_d, wsd.model4_6_weight_d, wsd.model4_6_bias_d, wsd.model4_6_running_mean_d, wsd.model4_6_running_var_d, fsds[si].fm4_3_d, kns[si].BatchNorm2d_k[3]);
    //PRINTF_WITH_RANK("Block 4 done! (%f s)", timer_read(1));
    //DumpTensor("fm4_3.txt", fss[si].fm4_3, 3);

    /*
     * Block 5
     */
    //timer_reset(1); timer_start(1);
    Conv2d_CL(fss[si].fm4_3, *ws.model5_0_weight, *ws.model5_0_bias, fss[si].fm5_0, 1, 2, 2, true, fsds[si].fm4_3_d, wsd.model5_0_weight_d, wsd.model5_0_bias_d, fsds[si].fm5_0_d, kns[si].Conv2d_k[10], true);
    Conv2d_CL(fss[si].fm5_0, *ws.model5_2_weight, *ws.model5_2_bias, fss[si].fm5_1, 1, 2, 2, true, fsds[si].fm5_0_d, wsd.model5_2_weight_d, wsd.model5_2_bias_d, fsds[si].fm5_1_d, kns[si].Conv2d_k[11], true);
    Conv2d_CL(fss[si].fm5_1, *ws.model5_4_weight, *ws.model5_4_bias, fss[si].fm5_2, 1, 2, 2, true, fsds[si].fm5_1_d, wsd.model5_4_weight_d, wsd.model5_4_bias_d, fsds[si].fm5_2_d, kns[si].Conv2d_k[12], true);
    BatchNorm2d_CL(fss[si].fm5_2, *ws.model5_6_weight, *ws.model5_6_bias, *ws.model5_6_running_mean, *ws.model5_6_running_var, fss[si].fm5_3, 1e-5, fsds[si].fm5_2_d, wsd.model5_6_weight_d, wsd.model5_6_bias_d, wsd.model5_6_running_mean_d, wsd.model5_6_running_var_d, fsds[si].fm5_3_d, kns[si].BatchNorm2d_k[4]);
    //PRINTF_WITH_RANK("Block 5 done! (%f s)", timer_read(1));
    //DumpTensor("fm5_3.txt", fss[si].fm5_3, 3);

    /*
     * Block 6
     */
    //timer_reset(1); timer_start(1);
    Conv2d_CL(fss[si].fm5_3, *ws.model6_0_weight, *ws.model6_0_bias, fss[si].fm6_0, 1, 2, 2, true, fsds[si].fm5_3_d, wsd.model6_0_weight_d, wsd.model6_0_bias_d, fsds[si].fm6_0_d, kns[si].Conv2d_k[13], true);
    Conv2d_CL(fss[si].fm6_0, *ws.model6_2_weight, *ws.model6_2_bias, fss[si].fm6_1, 1, 2, 2, true, fsds[si].fm6_0_d, wsd.model6_2_weight_d, wsd.model6_2_bias_d, fsds[si].fm6_1_d, kns[si].Conv2d_k[14], true);
    Conv2d_CL(fss[si].fm6_1, *ws.model6_4_weight, *ws.model6_4_bias, fss[si].fm6_2, 1, 2, 2, true, fsds[si].fm6_1_d, wsd.model6_4_weight_d, wsd.model6_4_bias_d, fsds[si].fm6_2_d, kns[si].Conv2d_k[15], true);
    BatchNorm2d_CL(fss[si].fm6_2, *ws.model6_6_weight, *ws.model6_6_bias, *ws.model6_6_running_mean, *ws.model6_6_running_var, fss[si].fm6_3, 1e-5, fsds[si].fm6_2_d, wsd.model6_6_weight_d, wsd.model6_6_bias_d, wsd.model6_6_running_mean_d, wsd.model6_6_running_var_d, fsds[si].fm6_3_d, kns[si].BatchNorm2d_k[5]);
    //PRINTF_WITH_RANK("Block 6 done! (%f s)", timer_read(1));
    //DumpTensor("fm6_3.txt", fss[si].fm6_3, 3);

    /*
     * Block 7
     */
    //timer_reset(1); timer_start(1);
    Conv2d_CL(fss[si].fm6_3, *ws.model7_0_weight, *ws.model7_0_bias, fss[si].fm7_0, 1, 1, 1, true, fsds[si].fm6_3_d, wsd.model7_0_weight_d, wsd.model7_0_bias_d, fsds[si].fm7_0_d, kns[si].Conv2d_k[16], true);
    Conv2d_CL(fss[si].fm7_0, *ws.model7_2_weight, *ws.model7_2_bias, fss[si].fm7_1, 1, 1, 1, true, fsds[si].fm7_0_d, wsd.model7_2_weight_d, wsd.model7_2_bias_d, fsds[si].fm7_1_d, kns[si].Conv2d_k[17], true);
    Conv2d_CL(fss[si].fm7_1, *ws.model7_4_weight, *ws.model7_4_bias, fss[si].fm7_2, 1, 1, 1, true, fsds[si].fm7_1_d, wsd.model7_4_weight_d, wsd.model7_4_bias_d, fsds[si].fm7_2_d, kns[si].Conv2d_k[18], true);
    BatchNorm2d_CL(fss[si].fm7_2, *ws.model7_6_weight, *ws.model7_6_bias, *ws.model7_6_running_mean, *ws.model7_6_running_var, fss[si].fm7_3, 1e-5, fsds[si].fm7_2_d, wsd.model7_6_weight_d, wsd.model7_6_bias_d, wsd.model7_6_running_mean_d, wsd.model7_6_running_var_d, fsds[si].fm7_3_d, kns[si].BatchNorm2d_k[6]);
    //PRINTF_WITH_RANK("Block 7 done! (%f s)", timer_read(1));
    //DumpTensor("fm7_3.txt", fss[si].fm7_3, 3);

    /*
     * Block 8
     */
    //timer_reset(1); timer_start(1);
    ConvTranspose2dReLU_CL(fss[si].fm7_3, *ws.model8_0_weight, *ws.model8_0_bias, fss[si].fm8_0, 2, 1, fsds[si].fm7_3_d, wsd.model8_0_weight_d, wsd.model8_0_bias_d, fsds[si].fm8_0_d, kns[si].ConvTranspose2dReLU_k);
    Conv2d_CL(fss[si].fm8_0, *ws.model8_2_weight, *ws.model8_2_bias, fss[si].fm8_1, 1, 1, 1, true, fsds[si].fm8_0_d, wsd.model8_2_weight_d, wsd.model8_2_bias_d, fsds[si].fm8_1_d, kns[si].Conv2d_k[19], true);
    Conv2d_CL(fss[si].fm8_1, *ws.model8_4_weight, *ws.model8_4_bias, fss[si].fm8_2, 1, 1, 1, true, fsds[si].fm8_1_d, wsd.model8_4_weight_d, wsd.model8_4_bias_d, fsds[si].fm8_2_d, kns[si].Conv2d_k[20], true);
    Conv2d_CL(fss[si].fm8_2, *ws.model8_6_weight, *ws.model8_6_bias, fss[si].fm8_3, 1, 0, 1, true, fsds[si].fm8_2_d, wsd.model8_6_weight_d, wsd.model8_6_bias_d, fsds[si].fm8_3_d, kns[si].Conv2d_k[21], false);
    //PRINTF_WITH_RANK("Block 8 done! (%f s)", timer_read(1));
    //DumpTensor("fm8_3.txt", fss[si].fm8_3, 3);

    /*
     * Wrap-up block
     */
    //timer_reset(1); timer_start(1);
    Softmax_CL(fss[si].fm8_3, fss[si].fm_softmax, fsds[si].fm8_3_d, fsds[si].fm_softmax_d, kns[si].Softmax_k);
    Conv2d_CL(fss[si].fm_softmax, *ws.model_out_weight, {}, fss[si].fm_model_out, 1, 0, 1, false, fsds[si].fm_softmax_d, wsd.model_out_weight_d, NULL, fsds[si].fm_model_out_d, kns[si].Conv2d_k[22], false);
    Upsample_CL(fss[si].fm_model_out, fss[si].fm_upsample4, 4, fsds[si].fm_model_out_d, fsds[si].fm_upsample4_d, kns[si].Upsample_k);
    UnnormalizeAB_CL(fss[si].fm_upsample4, image_AB, fsds[si].fm_upsample4_d, fsds[si].image_AB_d, kns[si].UnnormalizeAB_k);
    //PRINTF_WITH_RANK("Block output done! (%f s)", timer_read(1));
    //DumpTensor("image_AB.txt", image_AB, 3);
}

void ColorizerFinalize() {
  // Free buffers we allocated in ColorizerInit
  for(int i=0;i<NUM_TDS;i++) {
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

/*
 * Make a new 3D Tensor. Caller is responsible to free its buf.
 */
Tensor Make3DTensor(int C, int H, int W) {
  return Tensor{(float*)malloc(C * H * W * sizeof(float)), {C, H, W}};
}

/*
 * Dump all contents of tensor to file. May help your debugging.
 */
/*
void DumpTensor(const char* filename, Tensor input, int dim) {
  FILE* f = fopen(filename, "w");
  if (dim == 3) {
    int C = input.shape[0], H = input.shape[1], W = input.shape[2];
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          fprintf(f, "[%d,%d,%d]=%f\n", c, h, w, input.buf[c * H * W + h * W + w]);
        }
      }
    }
  } else {
    CHECK_ERROR(false, "unexpected dimension");
  }
  fclose(f);
}
*/

/*
 * Normalize L channel.
 * Formula: y = (x - 50) / 100
 */
void NormalizeL(Tensor input, Tensor output) {
  int H = input.shape[1], W = input.shape[2];
  CHECK_ERROR(input.shape[0] == 1 && output.shape[0] == 1 && output.shape[1] == H && output.shape[2] == W, "Size mismatch");

  for (int h = 0; h < H; ++h) {
    for (int w = 0; w < W; ++w) {
      output.buf[h * W + w] = (input.buf[h * W + w] - 50) / 100;
    }
  }
}

void NormalizeL_CL(Tensor input, Tensor output, cl_mem input_d, cl_mem output_d, cl_kernel kernel, int si) {
  int H = input.shape[1], W = input.shape[2];
  CHECK_ERROR(input.shape[0] == 1 && output.shape[0] == 1 && output.shape[1] == H && output.shape[2] == W, "Size mismatch");

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_d);  
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_d);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 2, sizeof(int), &H);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 3, sizeof(int), &W);
  CHECK_ERROR_CL(err);

  size_t gws[2] = {(size_t)H, (size_t)W}, lws[2] = {1, 1};
  for (int i = 0; i < 2; ++i) {
    gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
  }

  err = clEnqueueNDRangeKernel(exec_queue, kernel, 2, NULL, gws, lws, 1, &write_input_e[si], NULL);
  CHECK_ERROR_CL(err);

  // err = clFinish(exec_queue);
  // CHECK_ERROR_CL(err);

  // err = clEnqueueReadBuffer(exec_queue, output_d, CL_TRUE, 0, 1 * H * W * sizeof(float), output.buf, 0, NULL, NULL);
  // CHECK_ERROR_CL(err);
}

/*
 * Convolution
 * input shape = (C, H, W)
 * weight shape = (K, C, R, S)
 * bias shape = (K)
 * output shape = (K, OH, OW)
 *   where OH = (H + 2 * pad - dilation * (R - 1) - 1) / stride + 1,
 *         OW = (W + 2 * pad - dilation * (S - 1) - 1) / stride + 1
 */
void Conv2d(Tensor input, Tensor weight, Tensor bias, Tensor output, int stride, int pad, int dilation, bool has_bias) {
  int C = input.shape[0], H = input.shape[1], W = input.shape[2];
  int K = weight.shape[0], R = weight.shape[2], S = weight.shape[3];
  int OH = output.shape[1], OW = output.shape[2];
  CHECK_ERROR(OH == (H + 2 * pad - dilation * (R - 1) - 1) / stride + 1, "Output height mismatch");
  CHECK_ERROR(OW == (W + 2 * pad - dilation * (S - 1) - 1) / stride + 1, "Output width mismatch");
  CHECK_ERROR(weight.shape[1] == C && (!has_bias || bias.shape[0] == K) && output.shape[0] == K, "Channel size mismatch");

  for (int k = 0; k < K; ++k) {
    for (int oh = 0; oh < OH; ++oh) {
      for (int ow = 0; ow < OW; ++ow) {
        float o = has_bias ? bias.buf[k] : 0;
        for (int c = 0; c < C; ++c) {
          for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
              int h = oh * stride - pad + r * dilation;
              int w = ow * stride - pad + s * dilation;
              if (h < 0 || h >= H || w < 0 || w >= W) continue;
              float i = input.buf[c * H * W + h * W + w];
              float f = weight.buf[k * C * R * S + c * R * S + r * S + s];
              o += i * f;
            }
          }
        }
        output.buf[k * OH * OW + oh * OW + ow] = o;
      }
    }
  }
}

void Conv2d_CL(Tensor input, Tensor weight, Tensor bias, Tensor output, int stride, int pad, int dilation, bool has_bias, cl_mem input_d, cl_mem weight_d, cl_mem bias_d, cl_mem output_d, cl_kernel kernel, bool with_relu) {
  int C = input.shape[0], H = input.shape[1], W = input.shape[2];
  int K = weight.shape[0], R = weight.shape[2], S = weight.shape[3];
  int OH = output.shape[1], OW = output.shape[2];
  CHECK_ERROR(OH == (H + 2 * pad - dilation * (R - 1) - 1) / stride + 1, "Output height mismatch");
  CHECK_ERROR(OW == (W + 2 * pad - dilation * (S - 1) - 1) / stride + 1, "Output width mismatch");
  CHECK_ERROR(weight.shape[1] == C && (!has_bias || bias.shape[0] == K) && output.shape[0] == K, "Channel size mismatch");

  // err = clEnqueueWriteBuffer(exec_queue, input_d, CL_FALSE, 0, C * H * W * sizeof(float), input.buf, 0, NULL, NULL);
  // CHECK_ERROR_CL(err);

  int has_bias_int = has_bias ? 1 : 0;
  int with_relu_int = with_relu ? 1 : 0;
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_d);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &weight_d);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias_d);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &output_d);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 4, sizeof(int), &stride);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 5, sizeof(int), &pad);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 6, sizeof(int), &dilation);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 7, sizeof(int), &has_bias_int);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 8, sizeof(int), &C);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 9, sizeof(int), &H);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 10, sizeof(int), &W);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 11, sizeof(int), &K);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 12, sizeof(int), &R);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 13, sizeof(int), &S);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 14, sizeof(int), &OH);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 15, sizeof(int), &OW);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 16, sizeof(int), &with_relu_int);
  CHECK_ERROR_CL(err);

  size_t gws[3] = {(size_t)K, (size_t)OH, (size_t)OW}, lws[3] = {1, 1, 1};
  for (int i = 0; i < 3; ++i) {
    gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
  }

  err = clEnqueueNDRangeKernel(exec_queue, kernel, 3, NULL, gws, lws, 0, NULL, NULL);
  CHECK_ERROR_CL(err);

  // err = clFinish(exec_queue);
  // CHECK_ERROR_CL(err);

  // err = clEnqueueReadBuffer(exec_queue, output_d, CL_TRUE, 0, K * OH * OW * sizeof(float), output.buf, 0, NULL, NULL);
  // CHECK_ERROR_CL(err);
}

/*
 * ReLU
 * Formula: y = max(x, 0)
 */
void ReLU(Tensor inout) {
  int C = inout.shape[0], H = inout.shape[1], W = inout.shape[2];
  for (int c = 0; c < C; ++c) {
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        int idx = c * H * W + h * W + w;
        inout.buf[idx] = inout.buf[idx] > 0 ? inout.buf[idx] : 0;
      }
    }
  }
}

/*
 * Batch Normaliztion
 * input shape = (C, H, W)
 * weight shape = (C)
 * bias shape = (C)
 * running_mean shape = (C)
 * running_var shape = (C)
 * output shape = (C, H, W)
 */
void BatchNorm2d(Tensor input, Tensor weight, Tensor bias, Tensor running_mean, Tensor running_var, Tensor output, const float eps) {
  int C = input.shape[0], H = input.shape[1], W = input.shape[2];
  CHECK_ERROR(weight.shape[0] == C && bias.shape[0] == C && running_mean.shape[0] == C && running_var.shape[0] == C, "Shape mismatch");
  CHECK_ERROR(output.shape[0] == C && output.shape[1] == H && output.shape[2] == W, "Shape mismatch");

  for (int c = 0; c < C; ++c) {
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        int idx = c * H * W + h * W + w;
        output.buf[idx] = (input.buf[idx] - running_mean.buf[c]) / sqrtf(running_var.buf[c] + eps) * weight.buf[c] + bias.buf[c];
      }
    }
  }
}

void BatchNorm2d_CL(Tensor input, Tensor weight, Tensor bias, Tensor running_mean, Tensor running_var, Tensor output, const float eps, cl_mem input_d, cl_mem weight_d, cl_mem bias_d, cl_mem running_mean_d, cl_mem running_var_d, cl_mem output_d, cl_kernel kernel) {
  int C = input.shape[0], H = input.shape[1], W = input.shape[2];
  CHECK_ERROR(weight.shape[0] == C && bias.shape[0] == C && running_mean.shape[0] == C && running_var.shape[0] == C, "Shape mismatch");
  CHECK_ERROR(output.shape[0] == C && output.shape[1] == H && output.shape[2] == W, "Shape mismatch");

  // err = clEnqueueWriteBuffer(exec_queue, input_d, CL_FALSE, 0, C * H * W * sizeof(float), input.buf, 0, NULL, NULL);
  // CHECK_ERROR_CL(err);

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_d);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &weight_d);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias_d);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &running_mean_d);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &running_var_d);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 5, sizeof(cl_mem), &output_d);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 6, sizeof(float), &eps);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 7, sizeof(int), &C);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 8, sizeof(int), &H);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 9, sizeof(int), &W);
  CHECK_ERROR_CL(err);

  size_t gws[3] = {(size_t)C, (size_t)H, (size_t)W}, lws[3] = {1, 1, 1};
  for (int i = 0; i < 3; ++i) {
    gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
  }

  err = clEnqueueNDRangeKernel(exec_queue, kernel, 3, NULL, gws, lws, 0, NULL, NULL);
  CHECK_ERROR_CL(err);

  // err = clFinish(exec_queue);
  // CHECK_ERROR_CL(err);

  // err = clEnqueueReadBuffer(exec_queue, output_d, CL_TRUE, 0, C * H * W * sizeof(float), output.buf, 0, NULL, NULL);
  // CHECK_ERROR_CL(err);
}

/*
 * Transposed convolution
 * input shape = (C, H, W)
 * weight shape = (C, K, R, S)
 * bias shape = (K)
 * output shape = (K, OH, OW)
 *   where OH = (H - 1) * stride - 2 * pad + R
 *         OW = (W - 1) * stride - 2 * pad + S
 */
void ConvTranspose2d(Tensor input, Tensor weight, Tensor bias, Tensor output, int stride, int pad) {
  int C = input.shape[0], H = input.shape[1], W = input.shape[2];
  int K = weight.shape[1], R = weight.shape[2], S = weight.shape[3];
  int OH = output.shape[1], OW = output.shape[2];

  CHECK_ERROR(OH == (H - 1) * stride - 2 * pad + R, "Output height mismatch");
  CHECK_ERROR(OW == (W - 1) * stride - 2 * pad + S, "Output width mismatch");
  CHECK_ERROR(weight.shape[0] == C && bias.shape[0] == K && output.shape[0] == K, "Channel size mismatch");

  for (int k = 0; k < K; ++k) {
    for (int oh = 0; oh < OH; ++oh) {
      for (int ow = 0; ow < OW; ++ow) {
        float o = bias.buf[k];
        for (int c = 0; c < C; ++c) {
          for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
              if ((oh + pad - r) % stride != 0) continue;
              if ((ow + pad - s) % stride != 0) continue;
              int h = (oh + pad - r) / stride;
              int w = (ow + pad - s) / stride;
              if (h < 0 || h >= H || w < 0 || w >= W) continue;
              float i = input.buf[c * H * W + h * W + w];
              float f = weight.buf[c * K * R * S + k * R * S + r * S + s];
              o += i * f;
            }
          }
        }
        output.buf[k * OH * OW + oh * OW + ow] = o;
      }
    }
  }
}

void ConvTranspose2dReLU_CL(Tensor input, Tensor weight, Tensor bias, Tensor output, int stride, int pad, cl_mem input_d, cl_mem weight_d, cl_mem bias_d, cl_mem output_d, cl_kernel kernel) {
  int C = input.shape[0], H = input.shape[1], W = input.shape[2];
  int K = weight.shape[1], R = weight.shape[2], S = weight.shape[3];
  int OH = output.shape[1], OW = output.shape[2];

  CHECK_ERROR(OH == (H - 1) * stride - 2 * pad + R, "Output height mismatch");
  CHECK_ERROR(OW == (W - 1) * stride - 2 * pad + S, "Output width mismatch");
  CHECK_ERROR(weight.shape[0] == C && bias.shape[0] == K && output.shape[0] == K, "Channel size mismatch");

  // err = clEnqueueWriteBuffer(exec_queue, input_d, CL_FALSE, 0, C * H * W * sizeof(float), input.buf, 0, NULL, NULL);
  // CHECK_ERROR_CL(err);

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_d);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &weight_d);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias_d);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &output_d);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 4, sizeof(int), &stride);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 5, sizeof(int), &pad);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 6, sizeof(int), &C);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 7, sizeof(int), &H);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 8, sizeof(int), &W);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 9, sizeof(int), &K);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 10, sizeof(int), &R);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 11, sizeof(int), &S);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 12, sizeof(int), &OH);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 13, sizeof(int), &OW);
  CHECK_ERROR_CL(err);

  size_t gws[3] = {(size_t)K, (size_t)OH, (size_t)OW}, lws[3] = {1, 1, 1};
  for (int i = 0; i < 3; ++i) {
    gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
  }

  err = clEnqueueNDRangeKernel(exec_queue, kernel, 3, NULL, gws, lws, 0, NULL, NULL);
  CHECK_ERROR_CL(err);

  // err = clFinish(exec_queue);
  // CHECK_ERROR_CL(err);

  // err = clEnqueueReadBuffer(exec_queue, output_d, CL_TRUE, 0, K * OH * OW * sizeof(float), output.buf, 0, NULL, NULL);
  // CHECK_ERROR_CL(err);
}

/*
 * Softmax
 * Formula: y = e^x / sum(e^x)
 */
void Softmax(Tensor input, Tensor output) {
  int C = input.shape[0], H = input.shape[1], W = input.shape[2];
  CHECK_ERROR(output.shape[0] == C && output.shape[1] == H && output.shape[2] == W, "shape mismatch");

  for (int h = 0; h < H; ++h) {
    for (int w = 0; w < W; ++w) {
      float sum = 0;
      for (int c = 0; c < C; ++c) {
        sum += expf(input.buf[c * H * W + h * W + w]);
      }
      for (int c = 0; c < C; ++c) {
        output.buf[c * H * W + h * W + w] = expf(input.buf[c * H * W + h * W + w]) / sum;
      }
    }
  }
}

void Softmax_CL(Tensor input, Tensor output, cl_mem input_d, cl_mem output_d, cl_kernel kernel) {
  int C = input.shape[0], H = input.shape[1], W = input.shape[2];
  CHECK_ERROR(output.shape[0] == C && output.shape[1] == H && output.shape[2] == W, "shape mismatch");

  // err = clEnqueueWriteBuffer(exec_queue, input_d, CL_FALSE, 0, C * H * W * sizeof(float), input.buf, 0, NULL, NULL);
  // CHECK_ERROR_CL(err);

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_d);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_d);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 2, sizeof(int), &C);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 3, sizeof(int), &H);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 4, sizeof(int), &W);
  CHECK_ERROR_CL(err);

  size_t gws[2] = {(size_t)H, (size_t)W}, lws[2] = {1, 1};
  for (int i = 0; i < 2; ++i) {
    gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
  }

  err = clEnqueueNDRangeKernel(exec_queue, kernel, 2, NULL, gws, lws, 0, NULL, NULL);
  CHECK_ERROR_CL(err);

  // err = clFinish(exec_queue);
  // CHECK_ERROR_CL(err);

  // err = clEnqueueReadBuffer(exec_queue, output_d, CL_TRUE, 0, C * H * W * sizeof(float), output.buf, 0, NULL, NULL);
  // CHECK_ERROR_CL(err);
}
/*
 * Bilinear interpolation
 * input shape = (C, H, W)
 * output shape = (C, floor(H * scale_factor), floor(W * scale_factor))
 */
void Upsample(Tensor input, Tensor output, float scale_factor) {
  int C = input.shape[0], H = input.shape[1], W = input.shape[2];
  int OH = output.shape[1], OW = output.shape[2];
  CHECK_ERROR(output.shape[0] == C && OH == floorf(H * scale_factor) && OW == floorf(W * scale_factor), "shape mismatch");

  for (int c = 0; c < C; ++c) {
    for (int oh = 0; oh < OH; ++oh) {
      for (int ow = 0; ow < OW; ++ow) {
        float h = (oh + 0.5) / scale_factor - 0.5;
        float w = (ow + 0.5) / scale_factor - 0.5;
        int h0 = floorf(h), w0 = floorf(w);
        int h1 = h0 + 1, w1 = w0 + 1;
        float h_offset = h - h0, w_offset = w - w0;
        float w00 = (1 - h_offset) * (1 - w_offset);
        float w01 = (1 - h_offset) * w_offset;
        float w10 = h_offset * (1 - w_offset);
        float w11 = h_offset * w_offset;
        h0 = h0 < 0 ? 0 : (h0 > H - 1 ? H - 1 : h0);
        h1 = h1 < 0 ? 0 : (h1 > H - 1 ? H - 1 : h1);
        w0 = w0 < 0 ? 0 : (w0 > W - 1 ? W - 1 : w0);
        w1 = w1 < 0 ? 0 : (w1 > W - 1 ? W - 1 : w1);
        output.buf[c * OH * OW + oh * OW + ow] = w00 * input.buf[c * H * W + h0 * W + w0]
                                               + w01 * input.buf[c * H * W + h0 * W + w1]
                                               + w10 * input.buf[c * H * W + h1 * W + w0]
                                               + w11 * input.buf[c * H * W + h1 * W + w1];
      }
    }
  }
}

void Upsample_CL(Tensor input, Tensor output, float scale_factor, cl_mem input_d, cl_mem output_d, cl_kernel kernel) {
  int C = input.shape[0], H = input.shape[1], W = input.shape[2];
  int OH = output.shape[1], OW = output.shape[2];
  CHECK_ERROR(output.shape[0] == C && OH == floorf(H * scale_factor) && OW == floorf(W * scale_factor), "shape mismatch");

  // err = clEnqueueWriteBuffer(exec_queue, input_d, CL_FALSE, 0, C * H * W * sizeof(float), input.buf, 0, NULL, NULL);
  // CHECK_ERROR_CL(err);

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_d);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_d);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 2, sizeof(float), &scale_factor);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 3, sizeof(int), &C);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 4, sizeof(int), &H);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 5, sizeof(int), &W);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 6, sizeof(int), &OH);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 7, sizeof(int), &OW);
  CHECK_ERROR_CL(err);

  size_t gws[3] = {(size_t)C, (size_t)OH, (size_t)OW}, lws[3] = {1, 1, 1};
  for (int i = 0; i < 3; ++i) {
    gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
  }

  err = clEnqueueNDRangeKernel(exec_queue, kernel, 3, NULL, gws, lws, 0, NULL, NULL);
  CHECK_ERROR_CL(err);

  // err = clFinish(exec_queue);
  // CHECK_ERROR_CL(err);

  // err = clEnqueueReadBuffer(exec_queue, output_d, CL_TRUE, 0, C * OH * OW * sizeof(float), output.buf, 0, NULL, NULL);
  // CHECK_ERROR_CL(err);
}

/*
 * Unnormalize A and B channel
 * Formula: y = x * 110
 */
void UnnormalizeAB(Tensor input, Tensor output) {
  int C = input.shape[0], H = input.shape[1], W = input.shape[2];
  CHECK_ERROR(output.shape[0] == C && output.shape[1] == H && output.shape[2] == W, "shape mismatch");

  for (int c = 0; c < C; ++c) {
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        output.buf[c * H * W + h * W + w] = input.buf[c * H * W + h * W + w] * 110;
      }
    }
  }
}

void UnnormalizeAB_CL(Tensor input, Tensor output, cl_mem input_d, cl_mem output_d, cl_kernel kernel) {
  int C = input.shape[0], H = input.shape[1], W = input.shape[2];
  CHECK_ERROR(output.shape[0] == C && output.shape[1] == H && output.shape[2] == W, "shape mismatch");

  // err = clEnqueueWriteBuffer(exec_queue, input_d, CL_FALSE, 0, C * H * W * sizeof(float), input.buf, 0, NULL, NULL);
  // CHECK_ERROR_CL(err);

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_d);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_d);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 2, sizeof(int), &C);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 3, sizeof(int), &H);
  CHECK_ERROR_CL(err);
  err = clSetKernelArg(kernel, 4, sizeof(int), &W);
  CHECK_ERROR_CL(err);

  size_t gws[3] = {(size_t)C, (size_t)H, (size_t)W}, lws[3] = {1, 1, 1};
  for (int i = 0; i < 3; ++i) {
    gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
  }

  err = clEnqueueNDRangeKernel(exec_queue, kernel, 3, NULL, gws, lws, 0, NULL, NULL);
  CHECK_ERROR_CL(err);

  err = clFinish(exec_queue);
  CHECK_ERROR_CL(err);

  err = clEnqueueReadBuffer(exec_queue, output_d, CL_TRUE, 0, C * H * W * sizeof(float), output.buf, 0, NULL, NULL);
  CHECK_ERROR_CL(err);
}

static void print_platform_info(cl_platform_id platform) {
  size_t sz;
  char *buf;
  CHECK_ERROR_CL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &sz));
  buf = (char*)malloc(sz);
  CHECK_ERROR_CL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, sz, buf, NULL));
  printf("Detected OpenCL platform: %s\n", buf);
  free(buf);
}

static void print_device_info(cl_device_id device) {
  size_t sz;
  char *buf;
  CHECK_ERROR_CL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &sz));
  buf = (char*)malloc(sz);
  CHECK_ERROR_CL(clGetDeviceInfo(device, CL_DEVICE_NAME, sz, buf, NULL));
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
  size_t just = fread(source_code, sizeof(char), source_size, file); just--;
  source_code[source_size] = '\0';
  fclose(file);
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_code, &source_size, &err);
  CHECK_ERROR_CL(err);
  free(source_code);
  err = clBuildProgram(program, 1, &device, "", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    CHECK_ERROR_CL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
    char *log = (char*)malloc(log_size + 1);
    CHECK_ERROR_CL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL));
    log[log_size] = 0;
    printf("Compile error:\n%s\n", log);
    free(log);
  }
  CHECK_ERROR_CL(err);
  return program;
}
