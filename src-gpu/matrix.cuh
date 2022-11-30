#pragma once

#include <vector>

#include "error_check.cuh"

template <typename T_ELEM>
int loadMMSparseMatrix(char *filename, char elem_type, bool csrFormat, int *m,
                       int *n, int *nnz, T_ELEM **aVal, int **aRowInd,
                       int **aColInd, int extendSymMatrix);
template <typename T>
struct CSRMatDevice
{
  CSRMatDevice(int row_size, int col_size, int nnz_size)
      : m_row_size(row_size),
        m_col_size(col_size),
        m_nnz(nnz_size)
  {
    CUDAERR(cudaMallocManaged(&m_d_rowptr, (m_row_size + 1) * sizeof(T)));
    CUDAERR(cudaMallocManaged(&m_d_colidx, m_nnz * sizeof(T)));
    CUDAERR(cudaMallocManaged(&m_d_val, m_nnz * sizeof(T)));
  }

  CSRMatDevice(char *filepath)
  {
    // int rowsA = 0; /* number of rows of A */
    // int colsA = 0; /* number of columns of A */
    // int nnzA = 0;  /* number of nonzeros of A */
    int baseA = 0;

    int *h_csrRowPtrA = NULL;
    int *h_csrColIndA = NULL;
    T *h_csrValA = NULL;

    loadMMSparseMatrix<T>(filepath, 'd', true, &m_row_size,
                          &m_col_size, &m_nnz, &h_csrValA, &h_csrRowPtrA,
                          &h_csrColIndA, true);

    baseA = h_csrRowPtrA[0];  // baseA = {0,1}

    printf("%d, %d, %d\n", m_row_size, m_col_size, m_nnz);

    CUDAERR(cudaMallocManaged(&m_d_rowptr, (m_row_size + 1) * sizeof(T)));
    CUDAERR(cudaMallocManaged(&m_d_colidx, m_nnz * sizeof(T)));
    CUDAERR(cudaMallocManaged(&m_d_val, m_nnz * sizeof(T)));

    for (int i = 0; i < m_row_size + 1; ++i)
    {
      m_d_rowptr[i] = h_csrRowPtrA[i] - baseA;
    }

    for (int i = 0; i < m_nnz; ++i)
    {
      m_d_colidx[i] = h_csrColIndA[i] - baseA;
      m_d_val[i] = h_csrValA[i] - baseA;
    }
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
    CUDAERR(cudaMallocManaged(&m_d_colptr, (m_col_size + 1) * sizeof(T)));
    CUDAERR(cudaMallocManaged(&m_d_rowidx, m_nnz * sizeof(T)));
    CUDAERR(cudaMallocManaged(&m_d_val, m_nnz * sizeof(T)));
  }

  CSCMatDevice(char *filepath)
  {
    // int rowsA = 0; /* number of rows of A */
    // int colsA = 0; /* number of columns of A */
    // int nnzA = 0;  /* number of nonzeros of A */
    int baseA = 0;

    int *h_cscRowIdxA = NULL;
    int *h_cscColPtrA = NULL;
    T *h_cscValA = NULL;

    loadMMSparseMatrix<T>(filepath, 'd', false, &m_row_size,
                          &m_col_size, &m_nnz, &h_cscValA, &h_cscRowIdxA,
                          &h_cscColPtrA, true);

    baseA = h_cscColPtrA[0];  // baseA = {0,1}

    printf("%d, %d, %d\n", m_row_size, m_col_size, m_nnz);

    CUDAERR(cudaMallocManaged(&m_d_colptr, (m_row_size + 1) * sizeof(T)));
    CUDAERR(cudaMallocManaged(&m_d_rowidx, m_nnz * sizeof(T)));
    CUDAERR(cudaMallocManaged(&m_d_val, m_nnz * sizeof(T)));

    for (int i = 0; i < m_row_size + 1; ++i)
    {
      m_d_colptr[i] = h_cscColPtrA[i] - baseA;
    }

    for (int i = 0; i < m_nnz; ++i)
    {
      m_d_rowidx[i] = h_cscRowIdxA[i] - baseA;
      m_d_val[i] = h_cscValA[i] - baseA;
    }
  }

  int *m_d_colptr;
  int *m_d_rowidx;
  T *m_d_val;

  int m_row_size;
  int m_col_size;
  int m_nnz;
};

// Flora TODO:
void resize(int rows, int cols, int nnz)
{
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
