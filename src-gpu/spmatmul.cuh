#pragma once

#include "matrix.cuh"

// bottleneck: write on C is hard to parallelize
template <typename T>
__global__ void countNnzKernel(CSRMatDevice<T> A, CSRMatDevice<T> B, int *nnz_num)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < A.m_row_size)
  {
    int mask[B.m_row_size]; // TODO: can not dynamically allocate array

    int a_ci_s = A.m_d_rowptr[idx];
    int a_ci_e = A.m_d_rowptr[idx + 1];

    for (int i = a_ci_s; i < a_ci_e; ++i)
    {
      int b_ci_s = B.m_d_rowptr[A.m_d_colidx[i]];
      int b_ci_e = B.m_d_rowptr[A.m_d_colidx[i] + 1];
      for (int j = b_ci_s; j < b_ci_e; ++j)
      {
        int nz_idx = B.m_d_colidx[j];
        if (mask[nz_idx] != 1)
        {
          mask[nz_idx] = 1;
          atomicAdd(&nnz_num[0], 1);
        }
      }
    }
  }
}

template <typename T>
u_int countCsrCsrNnzHost(CSRMatDevice<T> A, CSRMatDevice<T> B)
{
  u_int nnz = 0;
  u_int mask[B.m_col_size];

  u_int a_rows = A.m_row_size;
  for (u_int m = 0; m < a_rows; ++m)
  {
    for (u_int jj = A.m_d_rowptr[m]; jj < A.m_d_rowptr[m + 1]; ++jj)
    {
      u_int j = A.m_d_colidx[jj];

      for (u_int kk = B.m_d_rowptr[j]; kk < B.m_d_rowptr[j + 1]; ++kk)
      {
        u_int k = B.m_d_colidx[kk];

        if (mask[k] != m)
        {
          mask[k] = m;
          nnz++;
        }
      }
    }
  }
  return nnz;
}

template <typename T>
__global__ void spgemmRowWiseMulKernel(CSRMatDevice<T> A, CSRMatDevice<T> B, T *c)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < A.m_row_size)
  {
    int a_ci_s = A.m_d_rowptr[idx];
    int a_ci_e = A.m_d_rowptr[idx + 1];

    for (int i = a_ci_s; i < a_ci_e; ++i)
    {
      T a_val = A.m_d_val[i];
      int b_ci_s = B.m_d_rowptr[A.m_d_colidx[i]];
      int b_ci_e = B.m_d_rowptr[A.m_d_colidx[i] + 1];
      for (int j = b_ci_s; j < b_ci_e; ++j)
      {
        T b_val = B.m_d_val[j];
        int col_idx = B.m_d_colidx[j];
        c[idx * A.m_row_size + col_idx] += a_val * b_val;
      }
    }
  }
}

template <typename T>
void spgemmRowWiseMul(CSRMatDevice<T> A, CSRMatDevice<T> B, T *c_arr)
{
  int t_num = 256;
  int b_num = (A.m_row_size + t_num - 1) / t_num;
  spgemmRowWiseMulKernel<<<b_num, t_num>>>(A, B, c_arr);
  cudaDeviceSynchronize();
}

// bottleneck: some threads get sum=0, waste of computation
template <typename T>
__global__ void spgemmInnProMulKernel(CSRMatDevice<T> A, CSCMatDevice<T> B, float *C)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int N = A.m_row_size;

  // printf("tid: %u, N: %u\n", tid, N);

  if (tid < N)
  {

    int csr_start = A.m_d_rowptr[tid];
    int csr_end = A.m_d_rowptr[tid + 1];
    int csc_start = B.m_d_colptr[tid];
    int csc_end = B.m_d_colptr[tid + 1];

    for (int n = 0; n < N; ++n)
    {
      // printf("Loop2: k: %u, n: %u\nA col idx: %u, B row idx: %u, \n\n\n",
      //       k,
      //       n,
      //       A.m_d_colidx[k],
      //       B.m_d_rowidx[n]
      //       );
      float temp_c = 0;

      for (int i = B.m_d_colptr[n]; i < B.m_d_colptr[n + 1]; ++i)
      {
        for (int k = csr_start; k < csr_end; ++k)
        {
          if (A.m_d_colidx[k] == B.m_d_rowidx[i])
          {
            temp_c += A.m_d_val[k] * B.m_d_val[i];
            // printf("\nTRUE: tid: %u,\nk: %u, n: %u, \nA col idx: %u, B row idx: %u, \nA val: %f, B val: %f, \nA * B: %f, c tmp: %f\n",
            //       tid,
            //       k,
            //       n,
            //       A.m_d_colidx[k],
            //       B.m_d_rowidx[i],
            //       A.m_d_val[k],
            //       B.m_d_val[i],
            //       A.m_d_val[k] * B.m_d_val[i],
            //       temp_c);
          }
        }
      }
      C[tid * N + n] = temp_c;
    }
  }
}

template <typename T>
__global__ void spgemmInnProMulKernel_v2(CSRMatDevice<T> A, CSCMatDevice<T> B, float *C)
{

  int N = A.m_row_size;

  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int csr_tid = tid / N;
  int csc_tid = tid % N;

  if (tid < N * N)
  {

    int csr_start = A.m_d_rowptr[csr_tid];
    int csr_end = A.m_d_rowptr[csr_tid + 1];
    int csc_start = B.m_d_colptr[csc_tid];
    int csc_end = B.m_d_colptr[csc_tid + 1];

    float temp_c = 0;

    for (int i = B.m_d_colptr[csc_tid]; i < B.m_d_colptr[csc_tid + 1]; ++i)
    {
      for (int k = csr_start; k < csr_end; ++k)
      {
        if (A.m_d_colidx[k] == B.m_d_rowidx[i])
        {
          temp_c += A.m_d_val[k] * B.m_d_val[i];
        }
      }
    }
    C[csr_tid * N + csc_tid] = temp_c;
  }
}

template <typename T>
void spgemmInnProMul(CSRMatDevice<T> A, CSCMatDevice<T> B, float *C)
{

  int t_num = 256;
  int b_num = (A.m_row_size + t_num - 1) / t_num;
  // spgemmInnProMulKernel<<<1, 8>>>(A, B, C);

  int b_num = (A.m_row_size * A.m_row_size + t_num - 1) / t_num;
  spgemmInnProMulKernel_v2<<<1, 16>>>(A, B, C);

  cudaDeviceSynchronize();
}

template <typename T>
__global__ void spgemmOutProMulKernel(CSCMatDevice<T> A, CSRMatDevice<T> B, float *C)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  int N = A.m_row_size;
  if (tid < N * N)
  {
    u_int csr_start = A.m_d_colptr[csc_tid];
    u_int csr_end = A.m_d_colptr[csc_tid + 1];
    u_int csc_start = B.m_d_rowptr[csr_tid];
    u_int csc_end = B.m_d_rowptr[csr_tid + 1];

    float sum = 0.0;

    // TODO: loop over B, atomic add C
    // TODO: input file output and store CSR/ CSC format
  }
}

template <typename T>
void spgemmOutProMul(CSRMatDevice<T> A, CSCMatDevice<T> B, float *C)
{

  dim3 dimGrid(1, 1);
  dim3 dimBlock(4, 4);

  cudaMallocManaged(&C, (A.m_row_size * A.m_row_size) * sizeof(float));
  spgemmOutProMulKernel<<<dimGrid, dimBlock>>>(A, B, C);
  cudaDeviceSynchronize();
}