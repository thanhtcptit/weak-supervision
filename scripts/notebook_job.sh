#!/bin/bash

#PBS -q DEFAULT
#PBS -oe
#PBS -l select=1
#PBS -N 2210421-ntb

cd /home/s2210421/projects/weak-supervision

module load singularity/3.9.5

singularity exec /home/s2210421/docker_images/dev jupyter notebook
