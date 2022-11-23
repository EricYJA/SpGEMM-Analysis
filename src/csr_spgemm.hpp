#include "csr_matrix.hpp"

/* row-wise product */
/*
mtx A (4 * 4)
row = [0, 2, 3, 5, 6]
val = [a, b,| c,| d, e,| f]
col = [0, 2,| 1,| 1, 3,| 3]

mtx B (4 * 4)
row = [0, 1, 2, 4, 6]
val = [a,| b,| c, d,| e, f]
col = [1,| 0,| 1, 2,| 1, 3]

*/


/* inner and outer product: CSR * CSC */