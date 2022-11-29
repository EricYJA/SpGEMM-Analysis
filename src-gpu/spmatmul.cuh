#pragma once

#include "matrix.cuh"

template <typename T>
__global__ void spgemmRowWiseMul(CSRMatDevice<T> a_mat, CSRMatDevice<T> b_mat) {
  
}

__global__ void spgemmInnProMul(CSRMatDevice A, CSCMatDevice B) {
    int csr_tid = threadIdx.x + blockDim.x * blockIdx.x;
    int csc_tid = threadIdx.y + blockDim.y * blockIdx.y;

    

}