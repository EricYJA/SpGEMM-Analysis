#pragma once

#include "matrix.cuh"

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
__global__ void spgemmRowWiseMulKernel(CSRMatDevice<T> A, CSRMatDevice<T> B, int *c)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < A.m_row_size)
  {
    int a_ci_s = A.m_d_rowptr[idx];
    int a_ci_e = A.m_d_rowptr[idx + 1];

    for (int i = a_ci_s; i < a_ci_e; ++i)
    {
      int a_val = A.m_d_val[i];
      int b_ci_s = B.m_d_rowptr[A.m_d_colidx[i]];
      int b_ci_e = B.m_d_rowptr[A.m_d_colidx[i] + 1];
      for (int j = b_ci_s; j < b_ci_e; ++j)
      {
        int b_val = B.m_d_val[j];
        int col_idx = B.m_d_colidx[j];
        c[idx * A.m_row_size + col_idx] = a_val * b_val;
      }
    }
  }
}

template <typename T>
void spgemmRowWiseMul(CSRMatDevice<T> A, CSRMatDevice<T> B, COOMatDevice<T> C)
{
  T* c_arr;
  cudaMallocManaged(&c_arr, A.m_row_size * B.m_col_size * sizeof(T));
}

// bottleneck: some threads get sum=0, waste of computation
template <typename T>
__global__ void spgemmInnProMulKernel(CSRMatDevice<T> A, CSCMatDevice<T> B, float* C) {
    int csr_tid = threadIdx.x + blockDim.x * blockIdx.x;
    int csc_tid = threadIdx.y + blockDim.y * blockIdx.y;

    printf("csr_tid: %u, csc_tid: %u\n",csr_tid, csc_tid);

    int N = A.m_row_size;
    T sum = 0.0;

    // printf("N: %u", N);
    if(csr_tid < N){

        int csr_start = A.m_d_rowptr[csr_tid];
        int csr_end = A.m_d_rowptr[csr_tid + 1];
        // int csr_range = A.m_d_rowptr[csr_tid + 1] - A.m_d_rowptr[csr_tid];
        int csr_range = csr_end - csr_start;
        printf("----csr_start: %u, csr_end: %u, A.m_d_rowptr[csr_tid + 1] - A.m_d_rowptr[csr_tid]: %u, csr_range: %u----\n",csr_start,csr_end,A.m_d_rowptr[csr_tid + 1] - A.m_d_rowptr[csr_tid], csr_range);

        int csc_start = B.m_d_colptr[csc_tid];
        int csc_end = B.m_d_colptr[csc_tid + 1];

        int csc_range = csc_end - csc_start;
        printf("----csc_start: %u, csc_end: %u, csc_range: %u----\n", csc_start, csc_end, csc_range);
        
        
        for(int k = 0; k < csr_range; ++k){
          for(int n = 0; n < csc_range; ++n){
            // printf("k: %u, csr_start: %u, csr_end: %u, csr_start - csr_end: %u, (A k, B n): (%u, %u)\nn: %u, csc_start: %u, csc_end: %u, csc_range: %u\n\n", k, csr_start, csr_end, csr_range, A.m_d_colidx[A.m_d_rowptr[csr_tid] + k],B.m_d_rowidx[B.m_d_colptr[csc_tid] + n], n, csc_start, csc_end, csc_range);

            if(A.m_d_colidx[csr_start + k] == B.m_d_rowidx[csc_start + n]){
                sum += A.m_d_val[csr_start + k] * B.m_d_val[csc_start + n];
                printf("A col idx: %u, B row idx: %u, A val idx: %u, A val: %f, B val idx: %u, B val: %f, A * B: %f\n",A.m_d_colidx[csr_start + k],B.m_d_rowidx[csc_start + n], csr_start + k, A.m_d_val[csr_start + k], csc_start + n, B.m_d_val[csc_start + n], A.m_d_val[csr_start + k] * B.m_d_val[csc_start + n]);
            }
          }
        }

    // C.m_d_val[csr_start] = sum;
    // C.m_d_colidx[csc_start] = csc_start;
    // printf("\nc val idx: %u, sum: %f, col idx: %u\n", csr_start, sum, csc_start);
    printf("sum: %f, csr_tid: %u, N: %u, csc_tid: %u, csr_tid * N + csc_tid: %u\n", sum, csr_tid, N, csc_tid, csr_tid * N + csc_tid);
    C[csr_tid * N + csc_tid] = sum;

    }
}

template <typename T>
void spgemmInnProMul(CSRMatDevice<T> A, CSCMatDevice<T> B, float* C)
{
  // countNnzKernel<<<1, 1>>>(A, B, C); // TODO
  // C.resize(A.m_row_size, A.m_col_size, C.nnz);
  // TODO: record time
  printf("start \n");
  dim3 dimGrid(1, 1);
  dim3 dimBlock(4, 4);

  spgemmInnProMulKernel<<<dimGrid, dimBlock>>>(A, B, C);
  cudaDeviceSynchronize();
}

template <typename T>
__global__ void spgemmOutProMul(CSCMatDevice<T> A, CSRMatDevice<T> B, COOMatDevice<T> C)
{
  int csr_tid = threadIdx.x + blockDim.x * blockIdx.x;
  int csc_tid = threadIdx.y + blockDim.y * blockIdx.y;

  int N = A.m_row_size;
  if (csr_tid < N)
  {
    u_int csr_start = A.m_d_colptr[csc_tid];
    u_int csr_end = A.m_d_colptr[csc_tid + 1];
    u_int csc_start = B.m_d_rowptr[csr_tid];
    u_int csc_end = B.m_d_rowptr[csr_tid + 1];

    double sum = 0.0;

    // TODO: loop over B, atomic add C
    // TODO: input file output and store CSR/ CSC format
  }
}