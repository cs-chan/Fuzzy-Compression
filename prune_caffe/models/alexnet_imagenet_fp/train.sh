#!/usr/bin/env sh

/home/william/caffe_pruning/build/tools/caffe train \
    --solver=models/alexnet_imagenet/solver.prototxt \
    --weights=models/alexnet_imagenet/alexnet_imagenet_fp.caffemodel
