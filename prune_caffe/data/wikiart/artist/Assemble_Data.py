import os
import shutil
import numpy as np


if __name__ == '__main__':
    HOME = os.environ['HOME']
    dataset_path = os.path.join(HOME, 'CaffeProjects/data/artist')
    source_path = os.path.join(HOME, 'PycharmProjects/Dataset/wikipainting/artist')

    if not os.path.exists(os.path.join(dataset_path, 'train')):
        os.mkdir(os.path.join(dataset_path, 'train'))
        os.mkdir(os.path.join(dataset_path, 'val'))
        train_f = open('train.txt', 'w')
        val_f = open('val.txt', 'w')

        images_path = os.path.join(source_path)
        print len(os.listdir(images_path))
        for cls, cls_name in enumerate(os.listdir(images_path)):
            cls_name_replaced = cls_name.replace(' ', '')
            os.mkdir(os.path.join(dataset_path, 'train', cls_name_replaced))
            os.mkdir(os.path.join(dataset_path, 'val', cls_name_replaced))
            file_names = os.listdir(os.path.join(images_path, cls_name))
            num_imgs = len(file_names)
            num_train = int(num_imgs * 0.66)

            train_imgs = np.random.permutation(num_imgs)
            val_imgs = train_imgs[num_train:]
            train_imgs = train_imgs[:num_train]

            for idx in train_imgs:
                shutil.copyfile(os.path.join(images_path, cls_name, file_names[idx]),
                                os.path.join(dataset_path, 'train', cls_name_replaced, file_names[idx]))
                train_f.write(os.path.join(cls_name_replaced, file_names[idx]) + ' ' + str(cls) + '\n')

            for idx in val_imgs:
                shutil.copyfile(os.path.join(images_path, cls_name, file_names[idx]),
                                os.path.join(dataset_path, 'val', cls_name_replaced, file_names[idx]))
                val_f.write(os.path.join(cls_name_replaced, file_names[idx]) + ' ' + str(cls) + '\n')

        train_f.close()
        val_f.close()
