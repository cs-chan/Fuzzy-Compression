#!/usr/bin/env sh

/home/william/caffe_pruning/build/tools/caffe test \
    --model=models/alexnet_finetune_genre_fp/train_val.prototxt \
    --weights=models/alexnet_finetune_genre_fp/caffe_alexnet_genre_train_fp.caffemodel \
    --iterations=50 --gpu=0
