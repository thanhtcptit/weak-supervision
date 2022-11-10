#!/bin/bash

#PBS -q SINGLE
#PBS -oe
#PBS -l select=1
#PBS -N 2210421-ws

cd /home/s2210421/projects/weak-supervision

module load singularity/3.9.5

singularity exec /home/s2210421/docker_images/dev ./scripts/run_bench.sh 10 train_logs/bench
