Important notes:
1. These codes were written around February 2016 using latest version of Nervana's Systems Neon library at that time.
2. Since then, many updates were released and the data structures were changed, e.g. dataloader is changed to Aeon, file structure for the saved model is different.
3. Hence, many errors will be encountered if these codes are used without proper modifications using the latest Neon version.
4. We do not have any intention to update the codes in Neon because we have decided to use Tensorflow instead for future works. 

Meanwhile, if older version Neon is installed, users may run the following files.
Note that we ran the codes in PyCharm. Hence, users may need to resolve the path error in the code.

## MNIST:

- To train from scratch, run the file: mnist_cnn.py
- To prune and re-train, run the file: mnist_cnn_fp.py

## CIFAR-10:
- To train from scratch: cifar10CNN
- To prune and re-train: cifar10CNNfp

