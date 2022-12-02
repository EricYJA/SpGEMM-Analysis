// #include "spm_array.h"
#pragma once

#include "mtx_triplet.hpp"
#include <algorithm>
#include <bits/stdc++.h>
#include <vector>
using namespace std;

typedef unsigned int u_int;
bool compare_row(const MTX_triplet &tri1, const MTX_triplet &tri2) {
  return (tri1.row_idx < tri2.row_idx);
};

bool is_zero(const MTX_triplet &temp) {
  double zero = 0.0;
  return (temp.val == zero);
};

class CSR_matrix {

public:
  u_int *csr_row;
  u_int *csr_col;
  double *csr_val;
  double *mtx;
  u_int row_num;
  u_int col_num;
  u_int nnz;

  /*csr constructor - empty matrix*/
  CSR_matrix() {}

  CSR_matrix(u_int r, u_int c) {
    row_num = r;
    col_num = c;
    csr_row = (u_int *)malloc(r * sizeof(u_int));
  }

  /* set value - array */
  void setFromArray(u_int *row, u_int *col, double *val) {

    csr_val = val;
    csr_row = row;
    csr_col = col;
  }

  /* set value - array */
  void setFromTriplets(std::vector<MTX_triplet> tri_vec) {
    // sort the vector by tri_vec.row
    // read all the triplets in the vector and put them into CSR_matrix.col, val
    // and use prefix sum for row

    // remove zero vals in the vector
    tri_vec.erase(std::remove_if(tri_vec.begin(), tri_vec.end(), is_zero),
                  tri_vec.end());

    u_int length = tri_vec.size();

    std::sort(tri_vec.begin(), tri_vec.end(), compare_row);

    csr_col = (u_int *)malloc(sizeof(u_int) * length);

    csr_val = (double *)malloc(sizeof(double) * length);
    u_int temp_row[length];

    for (int i = 0; i < length; ++i) {
      csr_col[i] = tri_vec[i].col_idx;
      csr_val[i] = tri_vec[i].val;
      temp_row[i] = tri_vec[i].row_idx;
    }

    csr_row = (u_int *)malloc(sizeof(u_int) * (row_num + 1));
    csr_row[0] = 0;
    int row_val = temp_row[0];
    *(csr_row + 1) = *temp_row;
    int itr = 1;

    // prefix sum of temp_row
    for (int i = 0; i < length; ++i) {
      if (row_val != temp_row[i]) {
        itr++;
        csr_row[itr] = csr_row[itr - 1];
        row_val = temp_row[i];
      }
      csr_row[itr]++;
    }

    int row_length = row_num + 1;
    nnz = csr_row[itr];
  }

  void resize(u_int r_n, u_int c_n, u_int n) {
    row_num = r_n;
    col_num = c_n;
    nnz = n;
    csr_col = (u_int *)malloc(nnz * sizeof(u_int));
    csr_row = (u_int *)malloc(row_num * sizeof(u_int));
    csr_val = (double *)malloc(nnz * sizeof(double));
  }
}