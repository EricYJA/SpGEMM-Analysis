# path="/home/ychenfei/research/sparse_matrix_perf_analysis/spgemm_dataflow_analysis/test_mtx2"
path="/data/toodemuy/datasets/SNAP"
dataflow="ip"
timefn=$dataflow
timefn+=_time_eval_SNAP.csv
memfn=$dataflow
memfn+=_mem_eval_SNAP.csv

# for f in $path/*.mtx; do
#     srun -p mario --gpus=1 src_gpu/./main $dataflow $f >> eval_result/$filename
# done

for f in $path/*.mtx; do
    srun -p mario --gpus=1 nvprof src_gpu/./main $dataflow $f --metrics dram_read_bytes,dram_write_bytes,dram_read_throughput,dram_write_throughput,dram_read_transactions,dram_write_transactions &>> eval_result/$memfn #mem eval
    # srun -p mario --gpus=1 nvprof src_gpu/./main $dataflow $f &>> eval_result/$timefn #run time eval

    # echo $f
done

    # srun -p mario --gpus=1 nvprof src_gpu/./main ip /data/toodemuy/datasets/SNAP/amazon0302.mtx --metrics dram_read_bytes,dram_write_bytes,dram_read_throughput,dram_write_throughput,dram_read_transactions,dram_write_transactions &>> eval_result/mem_test.csv
    # srun -p mario --gpus=1 nvprof src_gpu/./main ip /home/ychenfei/research/sparse_matrix_perf_analysis/spgemm_dataflow_analysis/test_mtx2/amazon0302.mtx --metrics dram_read_bytes,dram_write_bytes,dram_read_throughput,dram_write_throughput,dram_read_transactions,dram_write_transactions >> eval_result/mem_test.csv
