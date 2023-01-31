#pragma once

#include "spmatmul.cuh"

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

void testSetMatData(CSRMatDevice<float> &spmat,
                    std::vector<int> &a_rp_vec,
                    std::vector<int> &a_ci_vec,
                    std::vector<float> &a_va_vec)
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

void testSetMatData(CSCMatDevice<float> &spmat,
                    std::vector<int> &a_cp_vec,
                    std::vector<int> &a_ri_vec,
                    std::vector<float> &a_va_vec)
{
  for (int i = 0; i < a_cp_vec.size(); ++i)
  {
    spmat.m_d_colptr[i] = a_cp_vec[i];
  }

  for (int i = 0; i < a_ri_vec.size(); ++i)
  {
    spmat.m_d_rowidx[i] = a_ri_vec[i];
    spmat.m_d_val[i] = a_va_vec[i];
  }
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

  testMemKernel<<<1, 8>>>(A);
  cudaDeviceSynchronize();
}

void testCountSize(){
  CSRMatDevice<float> A(4, 4, 7);
  CSCMatDevice<float> B(4, 4, 7);

  std::vector<int> a_rp_vec = {0, 2, 4, 6, 7};
  std::vector<int> a_ci_vec = {0, 1, 1, 2, 0, 3, 2};
  std::vector<float> a_va_vec = {1.0, 4.0, 2.0, 3.0, 5.0, 7.0, 9.0};

  std::vector<int> b_cp_vec = {0, 2, 4, 6, 7};
  std::vector<int> b_ri_vec = {0, 2, 0, 1, 1, 3, 2};
  std::vector<float> b_va_vec = {1.0, 5.0, 4.0, 2.0, 3.0, 9.0, 7.0};

  testSetMatData(A, a_rp_vec, a_ci_vec, a_va_vec);
  testSetMatData(B, a_rp_vec, a_ci_vec, a_va_vec);

  float *c;
  cudaMallocManaged(&c, (A.m_row_size * A.m_row_size) * sizeof(float));

  int *count;
  cudaMallocManaged(&count, sizeof(int));


  spgemmInnProMul<float>(A, B, c);
//   spgemmSizeCount<float>(A, B, count);

  // printf("%d",count);

  for (int i = 0; i < 16; ++i)
  {
    printf("%f, ", c[i]);
  }
  printf("\n");
}

void testRowWise()
{
  std::vector<int> a_rp_vec = {0, 1, 2, 4};
  std::vector<int> a_ci_vec = {1, 2, 0, 1};
  std::vector<float> a_va_vec = {10, 11, 12, 13};

  CSRMatDevice<float> A(3, 3, 4);
  CSRMatDevice<float> B(3, 3, 4);

  testSetMatData(A, a_rp_vec, a_ci_vec, a_va_vec);
  testSetMatData(B, a_rp_vec, a_ci_vec, a_va_vec);

  CSRMatDevice<float> C((char *)"../TestMtx/cage3.mtx");
  CSRMatDevice<float> D((char *)"../TestMtx/cage3.mtx");

  printf("%d, %d, %d\n", C.m_d_rowptr[0], C.m_d_rowptr[1], C.m_d_rowptr[2]);

  int flat_size = C.m_row_size * D.m_col_size;
  printf("f_size: %d\n", flat_size);
  float *c_arr;
  cudaMallocManaged(&c_arr, flat_size * sizeof(float));
  spgemmRowWiseMul<float>(C, D, c_arr);
  for (int i = 0; i < flat_size; ++i)
  {
    printf("%f ", c_arr[i]);
  }
  printf("\n");
}

void testInnPro()
{
  CSRMatDevice<float> A(4, 4, 7);
  CSCMatDevice<float> B(4, 4, 7);

  std::vector<int> a_rp_vec = {0, 2, 4, 6, 7};
  std::vector<int> a_ci_vec = {0, 1, 1, 2, 0, 3, 2};
  std::vector<float> a_va_vec = {1.0, 4.0, 2.0, 3.0, 5.0, 7.0, 9.0};

  std::vector<int> b_cp_vec = {0, 2, 4, 6, 7};
  std::vector<int> b_ri_vec = {0, 2, 0, 1, 1, 3, 2};
  std::vector<float> b_va_vec = {1.0, 5.0, 4.0, 2.0, 3.0, 9.0, 7.0};

  testSetMatData(A, a_rp_vec, a_ci_vec, a_va_vec);
  testSetMatData(B, a_rp_vec, a_ci_vec, a_va_vec);

  float *c;
  cudaMallocManaged(&c, (A.m_row_size * A.m_row_size) * sizeof(float));

  spgemmInnProMul<float>(A, B, c);

  for (int i = 0; i < 16; ++i)
  {
    printf("%f, ", c[i]);
  }
  printf("\n");
}

void testOutPro()
{
  CSCMatDevice<float> A(4, 4, 7);
  CSRMatDevice<float> B(4, 4, 7);

  std::vector<int> a_rp_vec = {0, 2, 4, 6, 7};
  std::vector<int> a_ci_vec = {0, 1, 1, 2, 0, 3, 2};
  std::vector<float> a_va_vec = {1.0, 4.0, 2.0, 3.0, 5.0, 7.0, 9.0};

  std::vector<int> b_cp_vec = {0, 2, 4, 6, 7};
  std::vector<int> b_ri_vec = {0, 2, 0, 1, 1, 3, 2};
  std::vector<float> b_va_vec = {1.0, 5.0, 4.0, 2.0, 3.0, 9.0, 7.0};

  testSetMatData(A, a_rp_vec, a_ci_vec, a_va_vec);
  testSetMatData(B, b_cp_vec, b_ri_vec, b_va_vec);

  float *c;
  cudaMallocManaged(&c, (A.m_row_size * A.m_row_size) * sizeof(float));

  spgemmOutProMul<float>(A, B, c);

  for (int i = 0; i < 16; ++i)
  {
    printf("%f, ", c[i]);
  }
  printf("\n");
}
