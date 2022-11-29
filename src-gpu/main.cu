#include "matrix.cuh"
#include "spmatmul.cuh"
#include <vector>

void testNnz() {

}

int main() {
  u_int test = 3;
  CSRMatDevice<float> a_mat(test, test, test);


  /* test inner product*/
  CSRMatDevice<float> A(4, 4, 9);
  CSCMatDevice<float> B(4, 4, 9);
  CSRMatDevice<float> C(4, 4, 9);
  std::vector<u_int> a_rp_vec = {0,2,4,7,9};
  std::vector<u_int> a_ci_vec = {0,1,1,2,0,3,4,2,4};
  std::vector<float> a_va_vec = {1.0,4.0,2.0,3.0,5.0,7.0,8.0,9.0,6.0};

  std::vector<u_int> b_cp_vec = {0,2,4,6,7,9};
  std::vector<u_int> b_ri_vec = {0,2,0,1,1,3,2,2,3};
  std::vector<float> b_va_vec = {1.0,5.0,4.0,2.0,3.0,9.0,7.0,8.0,6.0};

  cudaMemcpy(A.m_d_rowptr, a_rp_vec.data(), a_rp_vec.size() * sizeof(float), cudaMemcpyHostToHost);
  A.m_d_rowptr = a_rp_vec.data();
  A.m_d_colidx = a_ci_vec.data();
  A.m_d_val = a_va_vec.data();

  B.m_d_colptr = b_cp_vec.data();
  B.m_d_rowidx = b_ri_vec.data();
  B.m_d_val = b_va_vec.data();

  printf("%u\n",A.m_d_rowptr[1]);
  printf("%u\n",A.m_d_colidx[8]);

  spgemmInnProMul<float>(A, B, C);

  printf("%u\n",C.m_d_rowptr[1]);
  for(int i = 0; i < 9; ++i){

  }
  printf("%u\n",C.m_d_colidx[2]);

  return 0;
}