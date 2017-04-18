#!/usr/bin/env sh

/home/william/caffe_pruning/build/tools/caffe train \
    --solver=models/alexnet_finetune_style_fp/solver.prototxt \
    --weights=models/alexnet_finetune_style_fp/alexnet_finetune_style_fp.caffemodel
