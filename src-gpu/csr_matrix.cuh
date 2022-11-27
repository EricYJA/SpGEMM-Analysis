#pragma once

#include <vector>

#include "error_check.cuh"

template <typename T>
struct CSRMatDevice
{
  CSRMatDevice(u_int row_size, u_int col_size, u_int nnz_size)
      : m_row_size(row_size),
        m_col_size(col_size),
        m_nnz(nnz_size)
  {
    CUDAERR(cudaMalloc(&m_d_rowptr, row_size * sizeof(T))));
    CUDAERR(cudaMalloc(&m_d_colidx, row_size * sizeof(T))));
    CUDAERR(cudaMalloc(&m_d_val, row_size * sizeof(T))));
  }

  u_int *m_d_rowptr;
  u_int *m_d_colidx;
  T *m_d_val;

  u_int m_row_size;
  u_int m_col_size;
  u_int m_nnz;
};