#include <vector>
#include <string>
#include <utility>
#include <cassert>
#include <iostream>

#include "spmatmul.cuh"
#include "spmattest.cuh"

void evalRowWise(char *filepath)
{
  printf("Use row-wise product dataflow\n");
  CSRMatDevice<float> A(filepath);
  CSRMatDevice<float> B(filepath);

  float *c_arr;
  int flat_size = A.m_row_size * B.m_col_size;
  cudaMallocManaged(&c_arr, flat_size * sizeof(float));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  spgemmRowWiseMul<float>(A, B, c_arr);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float gpu_elapsed_time_ms = 0;
  cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
  printf("GPU test Runing Time (ms): %f \n", gpu_elapsed_time_ms);
}

void evalInnProd(char *filepath)
{
  printf("Use inner product dataflow\n");
  CSRMatDevice<float> A(filepath);
  CSCMatDevice<float> B(filepath);

  float *c_arr;
  int flat_size = A.m_row_size * B.m_col_size;
  cudaMallocManaged(&c_arr, flat_size * sizeof(float));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  spgemmInnProMul<float>(A, B, c_arr);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float gpu_elapsed_time_ms = 0;
  cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
  printf("GPU test Runing Time (ms): %f \n", gpu_elapsed_time_ms);
}

void evalOutProd(char *filepath)
{
  printf("Use outer product dataflow\n");
  CSCMatDevice<float> A(filepath);
  CSRMatDevice<float> B(filepath);

  float *c_arr;
  int flat_size = A.m_row_size * B.m_col_size;
  cudaMallocManaged(&c_arr, flat_size * sizeof(float));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  spgemmOutProMul<float>(A, B, c_arr);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float gpu_elapsed_time_ms = 0;
  cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
  printf("GPU test Runing Time (ms): %f \n", gpu_elapsed_time_ms);
}

int main(int argc, const char *argv[])
{
  if (argc != 3)
  {
    std::cerr << "Usage: " << argv[0] << " <dataflow_type{ip, op, rw}> <matrix_path>\n";
    exit(2);
  }

  if (strcmp(argv[1], "ip") == 0)
  {
    evalInnProd((char *)argv[2]);
  }
  else if (strcmp(argv[1], "op") == 0)
  {
    evalOutProd((char *)argv[2]);
  }
  else if (strcmp(argv[1], "rw") == 0)
  {
    evalRowWise((char *)argv[2]);
  }
  else
  {
    std::cerr << "Unknown dataflow " << argv[1] << " \n";
    exit(1);
  }

  return 0;
}
