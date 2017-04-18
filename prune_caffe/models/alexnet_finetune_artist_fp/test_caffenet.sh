#!/usr/bin/env sh

/home/william/caffe_pruning/build/tools/caffe test \
    --model=models/alexnet_finetune_artist_fp/train_val.prototxt \
    --weights=models/alexnet_finetune_artist_fp/caffe_alexnet_artist_train_fp.caffemodel \
    --iterations=50 --gpu=0
