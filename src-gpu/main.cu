#include "matrix.cuh"
#include "spmatmul.cuh"
#include <vector>
// #include "testInnPro.h"


__global__ void testMemKernel(CSRMatDevice<float> spmat) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < spmat.m_row_size) {
    int s_i = spmat.m_d_rowptr[idx];
    int s_e = spmat.m_d_rowptr[idx + 1];
    for (int i= s_i; i < s_e; ++i) {
      printf("(%d, %d) ", spmat.m_d_colidx[i], spmat.m_d_val[i]);
    }
    printf("\n");
  }
}

void testSetMatData(CSRMatDevice<float>& spmat, std::vector<int>& a_rp_vec, std::vector<int>& a_ci_vec, std::vector<float>& a_va_vec) {
  for (int i = 0; i < a_rp_vec.size(); ++i) {
    spmat.m_d_rowptr[i] = a_rp_vec[i];
  }

  for (int i = 0; i < a_ci_vec.size(); ++i) {
    spmat.m_d_colidx[i] = a_ci_vec[i];
    spmat.m_d_val[i] = a_va_vec[i];
  }
}

void testNnz() {
  std::vector<int> a_rp_vec = {0,1,2,4};
  std::vector<int> a_ci_vec = {1,2,0,1};
  std::vector<float> a_va_vec = {10,11,12,13};

  CSRMatDevice<float> A(3, 3, 4);
  CSRMatDevice<float> B(3, 3, 4);

  testSetMatData(A, a_rp_vec, a_ci_vec, a_va_vec);
  testSetMatData(B, a_rp_vec, a_ci_vec, a_va_vec);

  testMemKernel<<<1, 8>>>(A);
  cudaDeviceSynchronize();
}

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

int main() {

  // testNnz();
  testInnPro();
  return 0;
}