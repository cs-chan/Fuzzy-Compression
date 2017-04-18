#!/usr/bin/env sh

/home/william/caffe_pruning/build/tools/caffe train \
    --solver=models/alexnet_finetune_genre_fp/solver.prototxt \
    --weights=models/alexnet_finetune_genre_fp/alexnet_finetune_genre_fp.caffemodel
