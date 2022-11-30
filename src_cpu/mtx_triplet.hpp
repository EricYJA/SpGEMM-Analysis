#pragma once
typedef unsigned int u_int;

class MTX_triplet{
    public:
        u_int row_idx;
        u_int col_idx;
        double val;

        MTX_triplet(){
            row_idx = 0;
            col_idx = 0;
            val = 0;
        };

        MTX_triplet(u_int x, u_int y, double z){
            row_idx = x;
            col_idx = y;
            val = z;
        };
};