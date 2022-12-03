# SpGEMM-Analysis
Analysis of the performance for different SPGEMM dataflow

### Dir Structure

```
.
├── eval_result // The evaluation result in csv format
├── README.md 
├── src_cpu // the cpu test&ref code
├── src_gpu // the core implementation
└── test_mtx // the test data
```

### Running Guide

```
$ cd src_gpu
$ make
$ ./main rw ../test_mtx/s100/nos4.mtx
$ make clean
```


