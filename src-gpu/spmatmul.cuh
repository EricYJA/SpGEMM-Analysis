#pragma once

#include "matrix.cuh"

template <typename T>
__global__ void countNnzKernel(CSRMatDevice<T> a_mat, CSRMatDevice<T> b_mat, int *nnz_num)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < a_mat.m_row_size)
  {
    u_int mask[b_mat.m_row_size];

    u_int a_ci_s = a_mat.m_d_rowptr[idx];
    u_int a_ci_e = a_mat.m_d_rowptr[idx + 1];

    for (int i = a_ci_s; i < a_ci_e; ++i)
    {
      u_int b_ci_s = b_mat.m_d_rowptr[a_mat.m_d_colidx[i]];
      u_int b_ci_e = b_mat.m_d_rowptr[a_mat.m_d_colidx[i] + 1];
      for (int j = b_ci_s; j < b_ci_e; ++j)
      {
        int nz_idx = b_mat.m_d_colidx[j];
        if (mask[nz_idx] != 1)
        {
          mask[k] = 1;
          atomicAdd(&nnz_num[0], 1);
        }
      }
    }
  }
}

template <typename T>
void spgemmRowWiseMul(CSRMatDevice<T> a_mat, CSRMatDevice<T> b_mat, COOMatDevice<T> c_mat)
{
  int *c_nnz;
  cudaMallocManaged(&c_nnz, sizeof(int));
  countNnzKernel(a_mat, b_mat, nnz_num);
}

__global__ spgemmInnProMul()
{
}