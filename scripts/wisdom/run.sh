#!/bin/bash

ROOT=$PWD

# arguments - dataset(1) mode(random/all/normal)(2) model(dt/lr/nn)(3) cardinality(4) num_of_loops(5)
#             save directory (6) #model_for_feature_keras (lstm/count/lemma) (7)

# cd $ROOT/modules/robust-aggregate-lfs/reef
# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 generic_generate_labels.py youtube normal dt 1 26 yt_val2.5_sup5_dt1 lemma

# CUDA_LAUNCH_BLOCKING=0 python3 gpu_rewt_ss_generic.py /tmp l1 0 l3 l4 0 l6 qg 5 <dataset_path> <num_class>
#      nn 0 <batch_size> <lr_learning_rate> <gm_learning_rate> normal f1

cd $ROOT/modules/robust-aggregate-lfs
LFS_DIR=/home/s2210421/projects/weak-supervision/modules/robust-aggregate-lfs/reef/LFs/youtube/yt_val2.5_sup5_dt1
NUM_CLASSES=2
CUDA_LAUNCH_BLOCKING=0 python3 gpu_rewt_ss_generic.py /tmp l1 0 l3 l4 0 l6 qg 5 $LFS_DIR \
    $NUM_CLASSES nn 0 32 0.0003 0.01 normal f1
