// #include "spm_array.h"
#pragma once

#include "mtx_triplet.hpp"
#include "csr_matrix.hpp"
#include <vector>
#include <bits/stdc++.h>
#include <algorithm>
using namespace std;

typedef unsigned int u_int;
// typedef spm_array s_a;
// typedef Eigen::Triplet<double> T;



/*row, col, val array*/
// struct csc_mtx{
//     u_int csc_val[0];
//     u_int csc_row[0];
//     double csc_col[0];
//     double mtx[0];
// }csc_mtx;

// std::vector<MTX_triplet> temp_vec;
// private:
// bool compare_row(const MTX_triplet& tri1, const MTX_triplet& tri2){
//     return (tri1.row_idx < tri2.row_idx);
// };

// bool is_zero(const MTX_triplet& temp){
//     double zero = 0.0;

//     printf("temp_val: %lf ",temp.val);
//     printf(temp.val == zero ? "true" : "false");
//     printf("\n");

//     return(temp.val == zero);
// };


class CSC_matrix{

    public:
        u_int* csc_row;
        u_int* csc_col;
        double* csc_val;
        double* mtx;
        u_int row_num;
        u_int col_num;
        u_int nnz = 0;

        // CSC_matrix();
        // CSC_matrix(u_int row_num, u_int col_num);
        // CSC_matrix(u_int* row, u_int* col, double* val);

        /*csc constructor - empty matrix*/
        CSC_matrix(){

        };

        CSC_matrix(u_int r, u_int c){
            row_num = r;
            col_num = c;
            mtx = (double*) malloc(r * c * sizeof(double));
            // mtx[r * c];
        }

        /* set value - array */
        void setFromArray(u_int* row, u_int* col, double* val){

            csc_val = val;
            csc_row = row;
            csc_col = col;

            /* if valid */

            // u_int nnz = row.back();
            // u_int num_rows = 
        };

        /* set value - array */
        void setFromTriplets(std::vector<MTX_triplet> tri_vec){
            // sort the vector by tri_vec.row
            // read all the triplets in the vector and put them into CSC_matrix.col, val and use prefix sum for row

            // remove zero vals in the vector

            printf("yayy \n\n");

            tri_vec.erase(
                std::remove_if(tri_vec.begin(), tri_vec.end(), is_zero),
                tri_vec.end());
            printf("\n");

            printf("tri no zero: \n");
            for(MTX_triplet i : tri_vec){
                printf("<%d, %d, %lf> \n", i.row_idx, i.col_idx, i.val);
            }
            printf("\n");


            u_int length = tri_vec.size();
            printf("tri_vec length: %d \n\n", length);

            printf("tri sort col: \n");
            std::sort(tri_vec.begin(), tri_vec.end(), compare_row);
            for(MTX_triplet i : tri_vec){
                printf("<%d, %d, %lf> \n", i.row_idx, i.col_idx, i.val);
            }
            printf("\n");

            // csc_col[length];
            csc_row = (u_int*) malloc(sizeof(u_int) * length);
            // csc_val[length];
            csc_val = (double*) malloc(sizeof(double) * length);
            u_int temp_col[length];

            for(int i = 0; i < length; ++i){
                csc_row[i] = tri_vec[i].row_idx;
                csc_val[i] = tri_vec[i].val;
                temp_col[i] = tri_vec[i].col_idx;
            }

            printf("temp col: \n");
            for(int i = 0; i < length; ++i){
                printf("%d ",temp_col[i]);
            }
            printf("\n");

            csc_col = (u_int*) malloc(sizeof(u_int) * (col_num + 1));
            csc_col[0] = 0;
            int col_val = temp_col[0];
            *(csc_col + 1) = *temp_col;
            int itr = 1;

            // prefix sum of temp_col
            for(int i = 0; i < length; ++i){
                if(col_val != temp_col[i]){
                    itr++;
                    csc_col[itr] = csc_col[itr - 1];
                    col_val = temp_col[i];
                }
                csc_col[itr]++;
            }

            int col_length = col_num + 1;
            printf("col ptr: \n");
            for(int i = 0; i < col_length; ++i){
                printf("%d ",csc_col[i]);
            }
            printf("\n");

            printf("row ptr: \n");
            for(int i = 0; i < length; ++i){
                printf("%d ",csc_row[i]);
            }
            printf("\n");

            printf("val ptr: \n");
            for(int i = 0; i < length; ++i){
                printf("%lf ",csc_val[i]);
            }
            printf("\n");

            nnz = csc_col[itr];
            printf("nnz: %d\n", nnz);

        };

        // print by row
        // void printMtx(CSC_matrix mtx){
        //     int col_nnz = 0;
        //     for(int c = 1; c < mtx.col_num; ++c){
        //         for(int r = 0; r < mtx.row_num; r += col_nnz){
        //             col_nnz = mtx.csc_col[c] - mtx.csc_col[c - 1];
        //             for(int n = 0; n < row_nnz; ++n){
        //                 // print()
        //             }
        //         }
        //     }
        // }


};

/*print csc matrix*/
// csc_print(csc_mtx mtx){

// }