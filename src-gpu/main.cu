#include "matrix.cuh"
#include "spmatmul.cuh"
#include <vector>

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

int main() {

  testNnz();
  return 0;
}