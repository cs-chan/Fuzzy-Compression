This repository contains code and trained models for the paper: 

Tan, W. R., Chan, C. S., Aguirre, H. E., & Tanaka, K. (2017). 
Fuzzy Quantitative Deep Compression Network. 
Neurocomputing, 2017.

Please cite this paper if you use this code as part of your published work. 

This repository requires CAFFE and/or Nervana Systems Neon to be installed. 
To install Nervana System Neon, please visit: https://github.com/NervanaSystems/neon
While for the CAFFE library, users have to contact the authors of the following paper to get the modified version CAFFE:

Han, Song, et al. "Learning both weights and connections for efficient neural network." 
Advances in Neural Information Processing Systems. 2015.

Codes for Wikiart dataset were written in CAFFE.
This repository does not include the Wikiart dataset used. 
Credit is given to the authors of the following paper for introducing the Wikiart dataset:

Saleh, B., & Elgammal, A. (2015). Large-scale Classification of Fine-Art Paintings: 
Learning The Right Metric on The Right Feature. arXiv preprint arXiv:1505.00855.

However, the dataset was not split into training and validation sets.
Users may access the split Wikiart dataset in the following website: http://www.cs-chan.com/publication.html
Note that the training and validation split in this website is not the same as the one used in the paper (we lost the original split due to hard disk failure). 
We do plan to update the codes to synchronize with the released split Wikiart dataset. 

For the rest of the datasets, please visit:
MNIST (codes written in Neon): http://yann.lecun.com/exdb/mnist/
CIFAR-10 (codes written in Neon): https://www.cs.toronto.edu/~kriz/cifar.html
ImageNet (codes written in CAFFE): http://image-net.org/challenges/LSVRC/2012/index

For more details, please read the readme.txt in the subdirectories.
We may release codes written in Tensorflow in future. 

Users may contact us at either of the following email for any question regarding our paper. 

Wei Ren, Tan: wrtan.edu at gmail.com
Chee Seng, Chan: cs.chan at um.edu.my 
