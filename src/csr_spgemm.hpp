#include "csr_matrix.hpp"
#include "csc_matrix.hpp"


/* row-wise product */
/*
mtx A (3 * 3)
row ptr: 0 2 4 5 
col ptr: 0 1 | 1 2 | 2 
val ptr: 1.000000 4.000000 | 2.000000 3.000000 | 9.000000 

mtx B (3 * 3)
row ptr: 0 2 4 5 
col ptr: 0 1 | 1 2 | 2 
val ptr: 1.000000 4.000000 | 2.000000 3.000000 | 9.000000 

*/

void spgemmRowWise(CSR_matrix A, CSR_matrix B, CSR_matrix Z){
    int row = A.row_num;
    int col = A.col_num;
    // int B_r = B.row_num;
    // int B_c = B.col_num;

    int a_c_itr = 0;

    int b_row_nnz[row];

    for(int i = 1; i < row + 1; ++row){
        b_row_nnz[i - 1] = A.csr_row[i] - A.csr_row[i - 1];
    }

    for(int a_r_itr = 1; a_r_itr < row + 1; ++a_r_itr){
        if(A.csr_row[a_r_itr] != A.csr_row[a_r_itr - 1]){
            int nnz_row = A.csr_row[a_r_itr] - A.csr_row[a_r_itr - 1];
            for(int i = a_c_itr; i < nnz_row; ++i){
                // for(int b_r_itr = 1; b_r_itr < row + 1; ++b_r_itr){

                // }
                int a = A.csr_val[a_c_itr];
                int a_c_idx = A.csr_col[a_c_itr];
                if(b_row_nnz[a_c_idx] != 0){
                    int b_r_idx = a_c_idx;
                    for(int j = b_r_idx; j < b_row_nnz[a_c_idx]; ++b_r_idx){
                        int b = B.csr_val[j];
                        Z[a_r_itr][] += a * b;
                    }
                }
                ++a_c_itr;
            }
        }
    }
}


/* inner and outer product: CSR * CSC */
/*
mtx A (3 * 3)
row ptr: 0 2 4 5 
col ptr: 0 1 | 1 2 | 2 
val ptr: 1.000000 4.000000 | 2.000000 3.000000 | 9.000000 

mtx B (3 * 3)
col ptr: 0 1 3 5 
row ptr: 0 | 0 1 | 1 2 
val ptr: 1.000000 4.000000 | 2.000000 3.000000 | 9.000000 

*/

/* return the nnz of Z*/

u_int spgemmNnzRowWise(CSR_matrix A, CSR_matrix B){
    
    u_int nnz = 0;
    u_int mask[B.col_num];

    u_int a_rows = A.row_num;
    for(u_int m = 0; m < a_rows; ++m){
        for(u_int jj = A.csr_row[m]; jj < A.csr_row[m + 1]; ++jj){
            u_int j = A.csr_col[jj];

            for(u_int kk = B.csr_row[j]; kk < B.csr_row[j + 1]; ++kk){
                u_int k = B.csr_col[kk];

                if(mask[k] != m){
                    mask[k] = m;
                    nnz++;
                }
            }
        }
    }
    return nnz;
}

u_int spgemmNnz(CSR_matrix A, CSC_matrix B){
    return 0;
}



void spgemmInnPro(CSR_matrix A, CSC_matrix B, CSR_matrix Z){
    // m*k k*n
    // loop through the rows of A(m)
    // get the values in the corresponding cols of B
    // compute Z

    u_int nnz = spgemmNnz(A, B);
    u_int a_rows = A.row_num;
    u_int b_cols = B.col_num;
    Z.row_num = a_rows;
    Z.col_num = b_cols;

    u_int itr = 0;

    for(u_int m = 0; m < a_rows; ++m){ //loop through the rows in A
        for(u_int j = A.csr_row[m]; j < A.csr_row[m + 1]; ++j){ //go over the nnz col idx of the current row
            u_int k = A.csr_col[j];
            double a_val = A.csr_val[j];

            for(u_int n = 0; n < b_cols; ++n){
                u_int z_val = 0;
                for(u_int i = B.csc_col[n]; i < B.csc_col[n + 1]; ++i){
                    if(k == B.csc_row[i]){
                        z_val += a_val * B.csc_val[i];                    
                    }
                }
                
                Z.csr_val[itr] = z_val;

            }

        }
    }
}