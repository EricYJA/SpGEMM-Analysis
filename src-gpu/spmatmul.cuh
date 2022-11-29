#pragma once

#include "matrix.cuh"

template <typename T>
__global__ void spgemmRowWiseMul(CSRMatDevice<T> a_mat, CSRMatDevice<T> b_mat) {
  
}

template <typename T>
__global__ void spgemmInnProMul(CSRMatDevice<T> A, CSCMatDevice<T> B, CSRMatDevice<T> C, u_int* temp_row) {
    int csr_tid = threadIdx.x + blockDim.x * blockIdx.x;
    int csc_tid = threadIdx.y + blockDim.y * blockIdx.y;

    int N = A.m_row_size;
    if(csr_tid < N){
        uint32_t csr_start = A.m_d_rowptr[csr_tid];
        uint32_t csr_end = A.m_d_row_ptr[csr_tid + 1];  
        uint32_t csc_start = B.m_d_col_ptr[csc_tid];
        uint32_t csc_end = B.m_d_col_ptr[csc_tid + 1];

        double sum = 0.0;
        
        for(int k = csr_tid; k < csr_start - csr_end; ++k){
            for(int n = 0; n < csc_start - csc_end; ++n){
                if(A.m_d_colidx[csr_start + k] == B.m_d_rowidx[csc_start + n]){
                    sum += A.m_d_val[csr_start + k] * B.m_d_val[csc_start + n];
                }
            }
        }

        C.m_d_val[csr_tid] = sum;
        C.m_d_colidx[csr_tid] = csc_start;
        temp_row[csr_tid] = csr_start;
    }
}

template <typename T>
void spgemmInnProMul(CSRMatDevice<T> A, CSCMatDevice<T> B, CSRMatDevice<T> C){ 
    int nnz = 0;
    u_int temp_row[nnz];
    C.resize(A.m_row_size, A.m_col_size, nnz);
    spgemmInnProMul<<<1, 1>>>(A, B, C, temp_row);
    C.getRowPtr(*temp_row);
}