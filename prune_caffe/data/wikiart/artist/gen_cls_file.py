import os
import numpy as np


if __name__ == '__main__':
    HOME = os.environ['HOME']
    dataset_path = os.path.join(HOME, 'CaffeProjects/data/artist')
    source_path = os.path.join(HOME, 'PycharmProjects/Dataset/wikipainting/artist')

    f = open('classes', 'w')

    images_path = os.path.join(source_path)
    print len(os.listdir(images_path))
    for cls, cls_name in enumerate(os.listdir(images_path)):
        cls_name_replaced = cls_name.replace(' ', '_')
        f.write(cls_name_replaced + ' ' + str(cls) + '\n')

    f.close()
