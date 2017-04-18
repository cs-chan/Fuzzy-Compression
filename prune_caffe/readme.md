Important notes:
1. After the paper was submitted, the hard disk that stored the reported experiment results was damaged and corrupted most of the data.
2. The experiments are re-conducted but the data re-collected are slightly different from the reported results in the paper due to different random seed used in the experiments. 
3. Note that the new results DO NOT CHANGE THE CONCLUSION of the work. 

To run the codes, run the following commands in the terminal with the correct path to the files:

## Wikiart dataset: 

For Artists:

To prune the weights:
```
python path/to/files/fp_artist.py
```
To train:
```
path/to/files/models/alexnet_finetune_artist_fp/train_caffenet.sh
```
To test trained model:
```
path/to/files/models/alexnet_finetune_artist_fp/test_caffenet.sh
```
For Genres:

To prune weights:
```
python path/to/files/fp_genre.py
```
To train:
```
path/to/files/models/alexnet_finetune_genre_fp/train_caffenet.sh
```
To test trained model:
```
path/to/files/models/alexnet_finetune_genre_fp/test_caffenet.sh
```
For Styles:

To prune weights:
```
python path/to/files/fp_style.py
```
To train:
```
path/to/files/models/alexnet_finetune_style_fp/train_caffenet.sh
```
To test trained model:
```
path/to/files/models/alexnet_finetune_style_fp/test_caffenet.sh
```

## ImageNet dataset:
* We are rushing for another journal, hence do not have time to re-run the experiment for ImageNet. Users may try to train their own pruned model for ImageNet with the provided codes.
** Note that different model requires different FQS settings to maximize the performance. 

To prune weights:
```
python path/to/files/fp_imagenet.py
```
To train:
```
path/to/files/models/alexnet_imagenet_fp/train_caffenet.sh
```
