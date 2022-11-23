#include "csr_matrix.hpp"

int main(){

    double val_arr[9] = {1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 8.0, 9.0, 6.0};
    u_int row_arr[5] = {0, 2, 4, 7, 9};
    u_int col_arr[9] = {0, 1, 1, 2, 0, 3, 4, 2, 4};

    u_int arr_r = 4;
    u_int arr_c = 4;

    CSR_matrix arr_mtx = CSR_matrix(arr_r, arr_c);
    // CSR_matrix emp_mtx;
    arr_mtx.setFromArray(row_arr, col_arr, val_arr);


    u_int tri_r = 3;
    u_int tri_c = 3;

    CSR_matrix tri_mtx = CSR_matrix(tri_r, tri_c);

    MTX_triplet tri1 = MTX_triplet(0, 0, 1.0);
    MTX_triplet tri2 = MTX_triplet(0, 1, 4.0);
    MTX_triplet tri3 = MTX_triplet(0, 2, 0.0);
    MTX_triplet tri4 = MTX_triplet(1, 0, 0.0);
    MTX_triplet tri5 = MTX_triplet(1, 1, 2.0);
    MTX_triplet tri6 = MTX_triplet(1, 2, 3.0);
    MTX_triplet tri7 = MTX_triplet(2, 0, 0.0);
    MTX_triplet tri8 = MTX_triplet(2, 1, 0.0);
    MTX_triplet tri9 = MTX_triplet(2, 2, 9.0);

    std::vector<MTX_triplet> tri_vec = {tri1, tri2, tri3, tri4, tri5, tri6, tri7, tri8, tri9};

    for(MTX_triplet i : tri_vec){
        printf("<%d, %d, %lf> \n", i.row_idx, i.col_idx, i.val);
    }

    printf("vector done (: \n");
    printf("test setFromTri:\n");
    tri_mtx.setFromTriplets(tri_vec);

}