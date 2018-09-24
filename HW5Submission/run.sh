#!/bin/bash

module load intel/2017A CUDA
module load CUDA
ulimit -S -s 1048576


echo "Submitting the job ..."

nvcc -arch=compute_35 -code=sm_35 -o par.exe main.cu
#./par.exe 10 3 3

bsub < grid.job


