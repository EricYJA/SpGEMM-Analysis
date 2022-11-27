// #include "spm_array.h"
#pragma once

#include "mtx_triplet.hpp"
#include <vector>
#include <bits/stdc++.h>
#include <algorithm>
using namespace std;

typedef unsigned int u_int;
// typedef spm_array s_a;
// typedef Eigen::Triplet<double> T;

/*row, col, val array*/
// struct csr_mtx{
//     u_int csr_val[0];
//     u_int csr_row[0];
//     double csr_col[0];
//     double mtx[0];
// }csr_mtx;

// std::vector<MTX_triplet> temp_vec;
// private:
bool compare_row(const MTX_triplet &tri1, const MTX_triplet &tri2)
{
    return (tri1.row_idx < tri2.row_idx);
};

bool is_zero(const MTX_triplet &temp)
{
    double zero = 0.0;

    printf("temp_val: %lf ", temp.val);
    printf(temp.val == zero ? "true" : "false");
    printf("\n");

    return (temp.val == zero);
};

class CSR_matrix
{

public:
    u_int *csr_row;
    u_int *csr_col;
    double *csr_val;
    double *mtx;
    u_int row_num;
    u_int col_num;
    u_int nnz;

    /*csr constructor - empty matrix*/
    CSR_matrix(){}

    CSR_matrix(u_int r, u_int c)
    {
        row_num = r;
        col_num = c;
        // mtx = (double*) malloc(r * c * sizeof(double));
        csr_row = (u_int *)malloc(r * sizeof(u_int));
        // mtx[r * c];
    }

    /* set value - array */
    void setFromArray(u_int *row, u_int *col, double *val)
    {

        csr_val = val;
        csr_row = row;
        csr_col = col;

        /* if valid */

        // u_int nnz = row.back();
        // u_int num_rows =
    }

    /* set value - array */
    void setFromTriplets(std::vector<MTX_triplet> tri_vec)
    {
        // sort the vector by tri_vec.row
        // read all the triplets in the vector and put them into CSR_matrix.col, val and use prefix sum for row

        // remove zero vals in the vector

        printf("yayy \n\n");

        tri_vec.erase(
            std::remove_if(tri_vec.begin(), tri_vec.end(), is_zero),
            tri_vec.end());
        printf("\n");

        printf("tri no zero: \n");
        for (MTX_triplet i : tri_vec)
        {
            printf("<%d, %d, %lf> \n", i.row_idx, i.col_idx, i.val);
        }
        printf("\n");

        u_int length = tri_vec.size();
        printf("tri_vec length: %d \n\n", length);

        printf("tri sort row: \n");
        std::sort(tri_vec.begin(), tri_vec.end(), compare_row);
        for (MTX_triplet i : tri_vec)
        {
            printf("<%d, %d, %lf> \n", i.row_idx, i.col_idx, i.val);
        }
        printf("\n");

        // csr_col[length];
        csr_col = (u_int *)malloc(sizeof(u_int) * length);
        // csr_val[length];
        csr_val = (double *)malloc(sizeof(double) * length);
        u_int temp_row[length];

        for (int i = 0; i < length; ++i)
        {
            csr_col[i] = tri_vec[i].col_idx;
            csr_val[i] = tri_vec[i].val;
            temp_row[i] = tri_vec[i].row_idx;
        }

        printf("temp row: \n");
        for (int i = 0; i < length; ++i)
        {
            printf("%d ", temp_row[i]);
        }
        printf("\n");

        csr_row = (u_int *)malloc(sizeof(u_int) * (row_num + 1));
        csr_row[0] = 0;
        int row_val = temp_row[0];
        *(csr_row + 1) = *temp_row;
        int itr = 1;

        // prefix sum of temp_row
        for (int i = 0; i < length; ++i)
        {
            if (row_val != temp_row[i])
            {
                itr++;
                csr_row[itr] = csr_row[itr - 1];
                row_val = temp_row[i];
            }
            csr_row[itr]++;
        }

        int row_length = row_num + 1;
        printf("row ptr: \n");
        for (int i = 0; i < row_length; ++i)
        {
            printf("%d ", csr_row[i]);
        }
        printf("\n");

        printf("col ptr: \n");
        for (int i = 0; i < length; ++i)
        {
            printf("%d ", csr_col[i]);
        }
        printf("\n");

        printf("val ptr: \n");
        for (int i = 0; i < length; ++i)
        {
            printf("%lf ", csr_val[i]);
        }
        printf("\n");

        nnz = csr_row[itr];
        printf("nnz: %d\n", nnz);
    }

    // print by row
    void printMtx(CSR_matrix mtx)
    {
        int row_nnz = 0;
        for (int r = 1; r < mtx.row_num; ++r)
        {
            for (int c = 0; c < mtx.col_num; c += row_nnz)
            {
                row_nnz = mtx.csr_row[r] - mtx.csr_row[r - 1];
                for (int n = 0; n < row_nnz; ++n)
                {
                    // print()
                }
            }
        }
    }

    void resize(u_int r_n, u_int c_n, u_int n)
    {
        row_num = r_n;
        col_num = c_n;
        nnz = n;
        csr_col = (u_int *)malloc(nnz * sizeof(u_int));
        csr_row = (u_int *)malloc(row_num * sizeof(u_int));
        csr_val = (double *)malloc(nnz * sizeof(double));
        // mtx =
    }
};

/*print csr matrix*/
// csr_print(csr_mtx mtx){

// }