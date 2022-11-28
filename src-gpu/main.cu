#include "csr_matrix.cuh"


int main() {
  u_int test = 3;
  CSRMatDevice<float> a_mat(test, test, test);
  
  return 0;
}