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


void spgemmRowWise(CSR_matrix A, CSR_matrix B, CSR_matrix Z){
    int row = A.row_num;
    int col = B.col_num;
    // int B_r = B.row_num;
    // int B_c = B.col_num;

    // TODO: Z mem pre-allocate - faster?
    // u_int nnz = spgemmNnzRowWise(A, B);
    // double z_val[nnz];
    // u_int z_col[nnz];

    std::map<std::pair<u_int, u_int>, double> z_map;

    // create triplet list
    // std::vector<MTX_triplet> Z_tri;


    for(int m = 0; m < row; ++m){
        for(int ii = A.csr_row[m]; ii < A.csr_row[m + 1]; ++ii){
            int k = A.csr_col[ii];
            double a_val = A.csr_val[ii];
            for(int jj = B.csr_row[k]; jj < B.csr_row[k + 1]; ++jj){
                // MTX_triplet tri = MTX_triplet();
                u_int n = B.csr_col[jj];
                double b_val = B.csr_val[jj];
                z_map[std::make_pair(m, n)].value += a_val * b_val;
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

void spgemmInnPro(CSR_matrix A, CSC_matrix B, CSR_matrix Z){
    // m*k k*n
    // loop through the rows of A(m)
    // get the values in the corresponding cols of B
    // compute Z

    // u_int nnz = spgemmNnz(A, B);
    u_int a_rows = A.row_num;
    u_int b_cols = B.col_num;
    // Z.row_num = a_rows;
    // Z.col_num = b_cols;
    std::map<std::pair<u_int, u_int>, double> z_map;


    // u_int itr = 0;

    for(u_int m = 0; m < a_rows; ++m){ //loop through the rows in A
        for(u_int j = A.csr_row[m]; j < A.csr_row[m + 1]; ++j){ //go over the nnz col idx of the current row
            u_int k = A.csr_col[j];
            double a_val = A.csr_val[j];

            for(u_int n = 0; n < b_cols; ++n){
                double z_val = 0.0;

                for(u_int i = B.csc_col[n]; i < B.csc_col[n + 1]; ++i){
                    if(k == B.csc_row[i]){ // a's col idx == b's row idx
                        // TODO: parallel
                        z_val += a_val * B.csc_val[i];                    
                    }
                }
                
                // Z.csr_val[itr] = z_val;
                z_map[std::make_pair(m, n)].value = z_val;


            }

        }
    }
}

void spgemmOutPro(CSC_matrix A, CSR_matrix B, CSR_matrix Z){
    // m*k k*n
    // loop through the rows of A(m)
    // get the values in the corresponding cols of B
    // compute Z

    // u_int nnz = spgemmNnz(A, B);
    u_int a_cols = A.col_num;
    u_int b_rows = B.row_num;
    // Z.row_num = a_rows;
    // Z.col_num = b_cols;
    std::map<std::pair<u_int, u_int>, double> z_map;


    // u_int itr = 0;

    for(u_int ak = 0; ak < a_cols; ++ak){
        for(u_int j = A.csc_col[ak]; j < A.csc_col[ak + 1]; ++j){
            
            u_int m = A.csc_row[j]; // a's row idx
            double a_val = A.csc_val[j];

            for(u_int bk = 0; bk < b_rows; ++bk){
                double z_val = 0.0;

                for(u_int i = B.csr_row[bk]; i < B.csr_row[bk + 1]; ++i){

                    if(m == B.csr_col[i]){ //a's row idx == b's col idx
                        // TODO: parallel
                        z_map[std::make_pair(ak, bk)].value += a_val * B.csr_val[i];
                    }
                }

                // z_map[std::make_pair(m, n)].value = z_val;
            }
        }
    }
}