// #include "spm_array.h"
#pragma once

#include "csr_matrix.hpp"
#include "mtx_triplet.hpp"
#include <algorithm>
#include <bits/stdc++.h>
#include <vector>
using namespace std;

typedef unsigned int u_int;
class CSC_matrix {

public:
  u_int *csc_row;
  u_int *csc_col;
  double *csc_val;
  double *mtx;
  u_int row_num;
  u_int col_num;
  u_int nnz = 0;

  /*csc constructor - empty matrix*/
  CSC_matrix(){

  };

  CSC_matrix(u_int r, u_int c) {
    row_num = r;
    col_num = c;
    mtx = (double *)malloc(r * c * sizeof(double));
  }

  /* set value - array */
  void setFromArray(u_int *row, u_int *col, double *val) {

    csc_val = val;
    csc_row = row;
    csc_col = col;
  };

  /* set value - array */
  void setFromTriplets(std::vector<MTX_triplet> tri_vec) {
    // sort the vector by tri_vec.row
    // read all the triplets in the vector and put them into CSC_matrix.col, val
    // and use prefix sum for row

    // remove zero vals in the vector
    tri_vec.erase(std::remove_if(tri_vec.begin(), tri_vec.end(), is_zero),
                  tri_vec.end());

    u_int length = tri_vec.size();

    std::sort(tri_vec.begin(), tri_vec.end(), compare_row);

    csc_row = (u_int *)malloc(sizeof(u_int) * length);

    csc_val = (double *)malloc(sizeof(double) * length);
    u_int temp_col[length];

    for (int i = 0; i < length; ++i) {
      csc_row[i] = tri_vec[i].row_idx;
      csc_val[i] = tri_vec[i].val;
      temp_col[i] = tri_vec[i].col_idx;
    }

    csc_col = (u_int *)malloc(sizeof(u_int) * (col_num + 1));
    csc_col[0] = 0;
    int col_val = temp_col[0];
    *(csc_col + 1) = *temp_col;
    int itr = 1;

    // prefix sum of temp_col
    for (int i = 0; i < length; ++i) {
      if (col_val != temp_col[i]) {
        itr++;
        csc_col[itr] = csc_col[itr - 1];
        col_val = temp_col[i];
      }
      csc_col[itr]++;
    }

    int col_length = col_num + 1;

    nnz = csc_col[itr];
  };
};