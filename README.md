# Neurocomputing 2017- Fuzzy Quantitative Deep Compression Network

Released on April, 2017

## Citation

This repository contains code and trained models for the paper: 

Tan, W. R., Chan, C. S., Aguirre, H. E., & Tanaka, K. (2017). 
Fuzzy Quantitative Deep Compression Network. 
Neurocomputing, 2017.

Please cite this paper if you use this code as part of your published work. 

## Description

This repository requires CAFFE and/or Nervana Systems Neon to be installed.
- To install Nervana System Neon, please visit: https://github.com/NervanaSystems/neon
- To install CAFFE library, users have to contact the authors of the following paper to get the modified CAFFE:

Han, S., Pool, J., Tran, J., & Dally, W. (2015). Learning both weights and connections for efficient neural network. In Advances in Neural Information Processing Systems (pp. 1135-1143).

## Datasets
Codes for Wikiart dataset are written in CAFFE.
This repository does not include the original Wikiart dataset used. 
Credit is given to the authors of the following paper for introducing the Wikiart dataset:

Saleh, B., & Elgammal, A. (2015). Large-scale Classification of Fine-Art Paintings: 
Learning The Right Metric on The Right Feature. arXiv preprint arXiv:1505.00855.

In order to replicate our work or have a fair comparison to our work, users may access the "new" Wikiart dataset (It was splitted into training and validation sets) in the following link: https://github.com/cs-chan/ICIP2016-PC/tree/master/WikiArt%20Dataset.

For the rest of the datasets, please visit:
- MNIST (codes written in Neon): http://yann.lecun.com/exdb/mnist/
- CIFAR-10 (codes written in Neon): https://www.cs.toronto.edu/~kriz/cifar.html
- ImageNet (codes written in CAFFE): http://image-net.org/challenges/LSVRC/2012/index

For more details, please read the readme files in the subdirectories.
We may release codes written in Tensorflow in the future. 

## Feedback
Suggestions and opinions of this work (both positive and negative) are greatly welcome. Please contact the authors by sending email to
`wrtan.edu at gmail.com`or `cs.chan at um.edu.my`.

## License
BSD-3, see `LICENSE` file for details.
