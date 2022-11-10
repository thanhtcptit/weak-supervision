#!/bin/bash

TYPE=$1

if [[ $TYPE = "bench" ]];
then
    bash -c "qsub -M thanhtc@jaist.ac.jp -m be ./scripts/benchmark_job.sh"
fi
