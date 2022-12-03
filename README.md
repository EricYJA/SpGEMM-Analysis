# SpGEMM-Analysis
Analysis of the performance for different SPGEMM dataflow

### Dir Structure

```
.
├── eval_result // The evaluation result in csv format
├── README.md 
├── src_cpu // the cpu test&ref code
├── src_gpu // the core implementation
│   ├── error_check.cuh 
│   ├── main.cu
│   ├── Makefile
│   ├── mmio.c // code from matrix market io
│   ├── mmio.h // code from matrix market io
│   ├── mmio_wrapper.cpp // code from cuda example
│   ├── run_ncu.sh
│   ├── spmat.cuh
│   ├── spmatmul.cuh
│   └── spmattest.cuh
└── test_mtx // the test data
```

### Running Guide

```
$ cd src_gpu
$ make
$ ./main rw ../test_mtx/s100/nos4.mtx
$ make clean
```


