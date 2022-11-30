#pragma once

#include "matrix.cuh"

template <typename T>
__global__ void countNnzKernel(CSRMatDevice<T> a_mat, CSRMatDevice<T> b_mat, int *nnz_num)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < a_mat.m_row_size)
  {
    int mask[b_mat.m_row_size]; // TODO: can not dynamically allocate array

    int a_ci_s = a_mat.m_d_rowptr[idx];
    int a_ci_e = a_mat.m_d_rowptr[idx + 1];

    for (int i = a_ci_s; i < a_ci_e; ++i)
    {
      int b_ci_s = b_mat.m_d_rowptr[a_mat.m_d_colidx[i]];
      int b_ci_e = b_mat.m_d_rowptr[a_mat.m_d_colidx[i] + 1];
      for (int j = b_ci_s; j < b_ci_e; ++j)
      {
        int nz_idx = b_mat.m_d_colidx[j];
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
void spgemmRowWiseMul(CSRMatDevice<T> a_mat, CSRMatDevice<T> b_mat, COOMatDevice<T> c_mat)
{
  int *c_nnz;
  cudaMallocManaged(&c_nnz, sizeof(int));
  countNnzKernel(a_mat, b_mat, c_nnz);
}

template <typename T>
__global__ void spgemmInnProMulKernel(CSRMatDevice<T> A, CSCMatDevice<T> B, CSRMatDevice<T> C)
{
  int csr_tid = threadIdx.x + blockDim.x * blockIdx.x;
  int csc_tid = threadIdx.y + blockDim.y * blockIdx.y;

  int N = A.m_row_size;
  if (csr_tid < N)
  {
    uint csr_start = A.m_d_rowptr[csr_tid];
    uint csr_end = A.m_d_rowptr[csr_tid + 1];
    uint csc_start = B.m_d_colptr[csc_tid];
    uint csc_end = B.m_d_colptr[csc_tid + 1];

    T sum = 0.0;

    for (int k = csr_tid; k < csr_start - csr_end; ++k)
    {
      for (int n = 0; n < csc_start - csc_end; ++n)
      {
        if (A.m_d_colidx[csr_start + k] == B.m_d_rowidx[csc_start + n])
        {
          sum += A.m_d_val[csr_start + k] * B.m_d_val[csc_start + n];
        }
      }
    }

    C.m_d_val[csr_tid] = sum;
    printf("sum: %f ", sum);
    C.m_d_colidx[csr_tid] = csc_start;
    printf("col idx: %u", csc_start);
  }
}

template <typename T>
void spgemmInnProMul(CSRMatDevice<T> A, CSCMatDevice<T> B, CSRMatDevice<T> C)
{
  // countNnzKernel<<<1, 1>>>(A, B, C); // TODO
  // C.resize(A.m_row_size, A.m_col_size, C.nnz);
  // TODO: record time
  spgemmInnProMulKernel<<<1, 1>>>(A, B, C);
}

template <typename T>
__global__ void spgemmOutProMul(CSCMatDevice<T> A, CSRMatDevice<T> B, COOMatDevice<T> C)
{
  int csr_tid = threadIdx.x + blockDim.x * blockIdx.x;
  int csc_tid = threadIdx.y + blockDim.y * blockIdx.y;

  int N = A.m_row_size;
  if (csr_tid < N)
  {
    uint csr_start = A.m_d_colptr[csc_tid];
    uint csr_end = A.m_d_colptr[csc_tid + 1];
    uint csc_start = B.m_d_rowptr[csr_tid];
    uint csc_end = B.m_d_rowptr[csr_tid + 1];

    double sum = 0.0;

    // TODO: loop over B, atomic add C
    // TODO: input file output and store CSR/ CSC format
  }
}