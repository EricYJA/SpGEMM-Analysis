#pragma once

#include "matrix.cuh"

__global__ spgemmRowWiseMul() {
  
}

__global__ void spgemmInnProMul(CSRMatDevice A, CSCMatDevice B) {
    int csr_tid = threadIdx.x + blockDim.x * blockIdx.x;
    int csc_tid = threadIdx.y + blockDim.y * blockIdx.y;

    

}