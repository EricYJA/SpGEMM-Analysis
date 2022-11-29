#pragma once

#include "matrix.cuh"

template <typename T>
__global__ void spgemmRowWiseNnzKernel(CSRMatDevice<T> a_mat, CSRMatDevice<T> b_mat, COOMatDevice<T> c_mat) {
  
}

template <typename T>
void spgemmRowWiseMul(CSRMatDevice<T> a_mat, CSRMatDevice<T> b_mat, COOMatDevice<T> c_mat) {
  u_int c_nnz = a_mat.m_nnz > b_mat.m_nnz ? a_mat.m_nnz : b_mat.m_nnz;
  
  
}

__global__ spgemmInnProMul() {
  
}