#include "matrix.cuh"
#include "spmatmul.cuh"
#include <vector>

 /* test inner product*/
 void testInnPro(){
  CSRMatDevice<float> A(4, 4, 7);
  CSCMatDevice<float> B(4, 4, 7);
  CSRMatDevice<float> C(4, 4, 49);
  std::vector<u_int> a_rp_vec = {0,2,4,6,7};
  std::vector<u_int> a_ci_vec = {0,1,1,2,0,3,4};
  std::vector<float> a_va_vec = {1.0,4.0,2.0,3.0,5.0,7.0,8.0};

  std::vector<u_int> b_cp_vec = {0,2,4,6,7};
  std::vector<u_int> b_ri_vec = {0,2,0,1,1,3,2};
  std::vector<float> b_va_vec = {1.0,5.0,4.0,2.0,3.0,9.0,7.0};

  // cudaMemcpy(A.m_d_rowptr, a_rp_vec.data(), (a_rp_vec.size()+1) * sizeof(u_int), cudaMemcpyHostToDevice);
  // cudaMemcpy(A.m_d_colidx, a_ci_vec.data(), 9 * sizeof(u_int), cudaMemcpyHostToHost);
  // cudaMemcpy(A.m_d_val, a_va_vec.data(), 9 * sizeof(float), cudaMemcpyHostToHost);

  // cudaMemcpy(B.m_d_colptr, b_cp_vec.data(), (b_cp_vec.size()+1) * sizeof(u_int), cudaMemcpyHostToHost);
  // cudaMemcpy(B.m_d_rowidx, b_ri_vec.data(), 9 * sizeof(u_int), cudaMemcpyHostToHost);
  // cudaMemcpy(B.m_d_val, b_va_vec.data(), 9 * sizeof(float), cudaMemcpyHostToHost);
  
  for(int i = 0; i<5; ++i){
    A.m_d_rowptr[i] = a_rp_vec[i];
  }

  for(int i = 0; i<7; ++i){
    A.m_d_colidx[i] = a_ci_vec[i];
    A.m_d_val[i] = a_va_vec[i];
  }

  for(int i = 0; i<5; ++i){
    B.m_d_colptr[i] = b_cp_vec[i];
  }

  for(int i = 0; i<7; ++i){
    B.m_d_rowidx[i] = b_ri_vec[i];
    B.m_d_val[i] = b_va_vec[i];
  }

  spgemmInnProMul<float>(A, B, C);


  printf("row ptr:\n");
  for(int i = 0; i < 5; ++i){
    printf("%u, ",C.m_d_rowptr[i]);
  }

  printf("\ncol idx:\n");

  for(int i = 0; i < 9; ++i){
    printf("%u,",C.m_d_colidx[i]);
  }
  printf("\n");
 }