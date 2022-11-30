#include <vector>

#include "matrix.cuh"
#include "spmatmul.cuh"

template <typename T_ELEM>
int loadMMSparseMatrix(char *filename, char elem_type, bool csrFormat, int *m,
                       int *n, int *nnz, T_ELEM **aVal, int **aRowInd,
                       int **aColInd, int extendSymMatrix);

__global__ void testMemKernel(CSRMatDevice<float> spmat)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < spmat.m_row_size)
  {
    int s_i = spmat.m_d_rowptr[idx];
    int s_e = spmat.m_d_rowptr[idx + 1];
    for (int i = s_i; i < s_e; ++i)
    {
      printf("(%d, %f) ", spmat.m_d_colidx[i], spmat.m_d_val[i]);
    }
  }
}

void testSetMatData(CSRMatDevice<float> &spmat, std::vector<int> &a_rp_vec, std::vector<int> &a_ci_vec, std::vector<float> &a_va_vec)
{
  for (int i = 0; i < a_rp_vec.size(); ++i)
  {
    spmat.m_d_rowptr[i] = a_rp_vec[i];
  }

  for (int i = 0; i < a_ci_vec.size(); ++i)
  {
    spmat.m_d_colidx[i] = a_ci_vec[i];
    spmat.m_d_val[i] = a_va_vec[i];
  }
}

void testload()
{
  int rowsA = 0; /* number of rows of A */
  int colsA = 0; /* number of columns of A */
  int nnzA = 0;  /* number of nonzeros of A */

  int *h_csrRowPtrA = NULL;
  int *h_csrColIndA = NULL;
  float *h_csrValA = NULL;

  loadMMSparseMatrix<float>("../TestMtx/cage3.mtx", 'd', true, &rowsA,
                            &colsA, &nnzA, &h_csrValA, &h_csrRowPtrA,
                            &h_csrColIndA, true);

  printf("%d, %d, %d\n", rowsA, colsA, nnzA);
}

void testNnz()
{
  std::vector<int> a_rp_vec = {0, 1, 2, 4};
  std::vector<int> a_ci_vec = {1, 2, 0, 1};
  std::vector<float> a_va_vec = {10, 11, 12, 13};

  CSRMatDevice<float> A(3, 3, 4);
  CSRMatDevice<float> B(3, 3, 4);

  testSetMatData(A, a_rp_vec, a_ci_vec, a_va_vec);
  testSetMatData(B, a_rp_vec, a_ci_vec, a_va_vec);

  int nnz_num = countCsrCsrNnzHost<float>(A, B);
  printf("%d\n", nnz_num);

  // int *nnz_num;
  // cudaMallocManaged(&nnz_num, sizeof(int));

  // countNnzKernel<float><<<1, 16>>>(A, B, nnz_num);
  // cudaDeviceSynchronize();

  // testMemKernel<<<1, 8>>>(A);
  // cudaDeviceSynchronize();
}

int main()
{

  testNnz();
  return 0;
}