#include "matrix.cuh"
#include "spmatmul.cuh"
#include <vector>
#include "testInnPro.h"

void testNnz() {

}

int main() {
  u_int test = 3;
  CSRMatDevice<float> a_mat(test, test, test);

  testInnPro();

  return 0;
}