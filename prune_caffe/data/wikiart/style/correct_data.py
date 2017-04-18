import os
import numpy as np


if __name__ == '__main__':
    HOME = os.environ['HOME']
    dataset_path = os.path.join(HOME, 'CaffeProjects/data/genre/train')
    # dataset_path = os.path.join(HOME, 'CaffeProjects/data/genre/val')

    train_f = open('train.txt', 'w')
    # val_f = open('val.txt', 'w')

    images_path = os.path.join(dataset_path)
    print len(os.listdir(images_path))
    for cls, cls_name in enumerate(os.listdir(images_path)):
        file_names = os.listdir(os.path.join(images_path, cls_name))
        num_imgs = len(file_names)

        for idx in range(num_imgs):
            train_f.write(os.path.join(cls_name, file_names[idx]) + ' ' + str(cls) + '\n')
            # val_f.write(os.path.join(cls_name, file_names[idx]) + ' ' + str(cls) + '\n')

    train_f.close()
    # val_f.close()
