#pragma once

#include <vector>

#include "error_check.cuh"

template <typename T>
struct CSRMatDevice
{
  CSRMatDevice(int row_size, int col_size, int nnz_size)
      : m_row_size(row_size),
        m_col_size(col_size),
        m_nnz(nnz_size)
  {
    CUDAERR(cudaMallocManaged(&m_d_rowptr, (m_row_size+1) * sizeof(T)));
    CUDAERR(cudaMallocManaged(&m_d_colidx, m_nnz * sizeof(T)));
    CUDAERR(cudaMallocManaged(&m_d_val, m_nnz * sizeof(T)));
  }

  int *m_d_rowptr;
  int *m_d_colidx;
  T *m_d_val;

  int m_row_size;
  int m_col_size;
  int m_nnz;

};


template <typename T>
struct CSCMatDevice
{
  CSCMatDevice(int row_size, int col_size, int nnz_size)
      : m_row_size(row_size),
        m_col_size(col_size),
        m_nnz(nnz_size)
  {
    CUDAERR(cudaMallocManaged(&m_d_colptr, (m_col_size+1) * sizeof(T)));
    CUDAERR(cudaMallocManaged(&m_d_rowidx, m_nnz * sizeof(T)));
    CUDAERR(cudaMallocManaged(&m_d_val, m_nnz * sizeof(T)));
  }

  int *m_d_colptr;
  int *m_d_rowidx;
  T *m_d_val;

  int m_row_size;
  int m_col_size;
  int m_nnz;
};

// Flora TODO:
void resize(int rows, int cols, int nnz){
    // cudaMallocManage
}


template <typename T>
struct COOMatDevice
{
  COOMatDevice(int row_size, int col_size, int nnz_size)
      : m_row_size(row_size),
        m_col_size(col_size),
        m_nnz(nnz_size)
  {
    CUDAERR(cudaMallocManaged(&m_d_colidx, m_nnz * sizeof(T)));
    CUDAERR(cudaMallocManaged(&m_d_rowidx, m_nnz * sizeof(T)));
    CUDAERR(cudaMallocManaged(&m_d_val, m_nnz * sizeof(T)));
  }

  int *m_d_rowidx;
  int *m_d_colidx;
  T *m_d_val;

  int m_row_size;
  int m_col_size;
  int m_nnz;
};
