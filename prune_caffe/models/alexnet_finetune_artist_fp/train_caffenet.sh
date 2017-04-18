#!/usr/bin/env sh

/home/william/caffe_pruning/build/tools/caffe train \
    --solver=models/alexnet_finetune_artist/solver.prototxt \
    --weights=models/alexnet_finetune_artist_fp/caffe_alexnet_artist_train_fp.caffemodel
