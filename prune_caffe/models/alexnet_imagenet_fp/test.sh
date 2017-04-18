#!/usr/bin/env sh

/home/william/caffe_pruning/build/tools/caffe test \
    --model=models/alexnet_imagenet/train_val.prototxt \
    --weights=models/alexnet_imagenet/caffe_alexnet_fp.caffemodel \
    --iterations=500 --gpu=0
