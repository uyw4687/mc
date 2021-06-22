#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <unistd.h>

#include "util.h"
#include "timer.h"
#include "colorizer.h"

int mpi_rank = 0, mpi_size = 1;

static char* input_filename;
static char* network_filename;
static char* output_filename;

static void PrintHelp(const char* prog_name);
static void ParseOpt(int argc, char** argv);

int main(int argc, char** argv) {
#ifdef USE_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

  ParseOpt(argc, argv);
  timer_init(9);

  float *input, *network, *output;
  int N;
  int H = 256, W = 256;

  BARRIER();

  if (mpi_rank == 0) {
    PRINTF_WITH_RANK("Reading input...");
    timer_reset(0); timer_start(0);
    size_t input_size;
    input = (float*)ReadFile(input_filename, &input_size);
    CHECK_ERROR(input_size % (H * W * sizeof(float)) == 0, "Input binary has unexpected size");
    N = input_size / (H * W * sizeof(float));
    PRINTF_WITH_RANK("Reading input done! (%d images, %f s)", N, timer_read(0));
  }

  BARRIER();

  if (mpi_rank == 0) {
    PRINTF_WITH_RANK("Reading network...");
    timer_reset(0); timer_start(0);
    size_t network_size;
    network = (float*)ReadFile(network_filename, &network_size);
    CHECK_ERROR(network_size == 128964012, "Network binary has unexpected size");
    PRINTF_WITH_RANK("Reading network done! (%ld bytes, %f s)", network_size, timer_read(0));
  }

  BARRIER();

  if (mpi_rank == 0) {
    PRINTF_WITH_RANK("Preparing output...");
    timer_reset(0); timer_start(0);
    size_t output_size = N * 2 * H * W * sizeof(float);
    output = (float*)malloc(output_size);
    CHECK_ERROR(output != NULL, "Failed to allocate memory for output");
    PRINTF_WITH_RANK("Preparing output done! (%f s)", timer_read(0));
  }

  BARRIER();

  PRINTF_WITH_RANK("Initializing...");
  timer_reset(0); timer_start(0);
  ColorizerInit();
  PRINTF_WITH_RANK("Initializing done! (%f s)", timer_read(0));

  BARRIER();

  PRINTF_WITH_RANK("Calculating...");
  timer_reset(0); timer_start(0);
  Colorize(input, network, output, N);

  BARRIER();

  double elapsed = timer_read(0);
  PRINTF_WITH_RANK("Calculating done! (%f s)", elapsed);
  PRINTF_WITH_RANK("Performance: %f img/s", N / elapsed);

  BARRIER();

  if (mpi_rank == 0) {
    PRINTF_WITH_RANK("Writing output...");
    timer_reset(0); timer_start(0);
    size_t output_size = N * 2 * H * W * sizeof(float);
    WriteFile(output_filename, output_size, output);
    PRINTF_WITH_RANK("Writing output done! (%f s)", timer_read(0));
  }

  BARRIER();

  PRINTF_WITH_RANK("Finalizing...");
  timer_reset(0); timer_start(0);
  ColorizerFinalize();
  PRINTF_WITH_RANK("Finalizing done! (%f s)", timer_read(0));

  BARRIER();

  if (mpi_rank == 0) {
    free(input);
    free(network);
    free(output);
  }

  timer_finalize();

#ifdef USE_MPI
  MPI_Finalize();
#endif

  return 0;
}

void PrintHelp(const char* prog_name) {
  if (mpi_rank == 0) {
    PRINTF_WITH_RANK("Usage: %s [input] [network] [output]", prog_name);
  }
}

void ParseOpt(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "h")) != -1) {
    switch (c) {
      case 'h':
        break;
      default:
        PrintHelp(argv[0]);
        EXIT(0);
    }
  }
  if (argc - optind != 3) {
    PrintHelp(argv[0]);
    EXIT(0);
  }
  input_filename = argv[optind + 0];
  network_filename = argv[optind + 1];
  output_filename = argv[optind + 2];
  if (mpi_rank == 0) {
    PRINTF_WITH_RANK("Options:");
    PRINTF_WITH_RANK("  Input: %s", input_filename);
    PRINTF_WITH_RANK("  Network: %s", network_filename);
    PRINTF_WITH_RANK("  Output: %s", output_filename);
    PRINTF_WITH_RANK("");
  }
}
