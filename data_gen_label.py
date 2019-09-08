# -*- coding: utf-8 -*-
import codecs
import math
import os
import random
from glob import glob

import keras
import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils, Sequence
from sklearn.model_selection import train_test_split
from random_eraser import get_random_eraser
import matplotlib.pyplot as plt
import pylab

def get_submodules_from_kwargs(kwargs):
    backend = keras.backend
    layers = keras.backend
    models = keras.models
    keras_utils = keras.utils

    return backend, layers, models, keras_utils


class BaseSequence(Sequence):
    """
    基础的数据流生成器，每次迭代返回一个batch
    BaseSequence可直接用于fit_generator的generator参数
    fit_generator会将BaseSequence再次封装为一个多进程的数据流生成器
    而且能保证在多进程下的一个epoch中不会重复取相同的样本
    """

    def __init__(self, img_paths, labels, batch_size, img_size, use, preprocess_input):
        assert len(img_paths) == len(labels), "len(img_paths) must equal to len(lables)"
        assert img_size[0] == img_size[1], "img_size[0] must equal to img_size[1]"
        self.x_y = np.hstack((np.array(img_paths).reshape(len(img_paths), 1), np.array(labels)))
        self.batch_size = batch_size
        self.img_size = img_size
        self.use = use
        self.preprocess_input = preprocess_input
        self.eraser = get_random_eraser( s_h=0.3,pixel_level=True)


    def __len__(self):
        return math.ceil(len(self.x_y) / self.batch_size)

    @staticmethod
    def center_img(img, size=None, fill_value=255):
        """
        center img in a square background
        """
        h, w = img.shape[:2]
        if size is None:
            size = max(h, w)
        shape = (size, size) + img.shape[2:]
        background = np.full(shape, fill_value, np.uint8)
        center_x = (size - w) // 2
        center_y = (size - h) // 2
        background[center_y:center_y + h, center_x:center_x + w] = img
        return background

    def preprocess_img(self, img_path):
        """
        image preprocessing
        you can add your special preprocess method here
        """
        img = Image.open(img_path)
        resize_scale = self.img_size[0] / max(img.size[:2])
        img = img.resize((int(img.size[0] * resize_scale), int(img.size[1] * resize_scale)))
        img = img.convert('RGB')
        img = np.array(img)

        # 数据增强
        if self.use:

            img = self.eraser(img)
            datagen = ImageDataGenerator(
                width_shift_range=0.05,
                height_shift_range=0.05,
                # rotation_range=90,
                # shear_range=0.1,
                # zoom_range=0.1,
                # brightness_range=(1, 1.3),
                horizontal_flip=True,
                vertical_flip=True,
            )
            img = datagen.random_transform(img)

        img = img[:, :, ::-1]
        img = self.center_img(img, self.img_size[0])
        # print(img)
        return img

    def __getitem__(self, idx):

        batch_x = self.x_y[idx * self.batch_size: (idx + 1) * self.batch_size, 0]
        batch_y = self.x_y[idx * self.batch_size: (idx + 1) * self.batch_size, 1:]

        batch_x = np.array([self.preprocess_img(img_path) for img_path in batch_x])
        batch_y = np.array(batch_y).astype(np.float32)
        # print(batch_y[1])

        # 获取归一化数据
        batch_x = self.preprocess_input(batch_x)

        return batch_x, batch_y

    def on_epoch_end(self):

        np.random.shuffle(self.x_y)


def smooth_labels(y, smooth_factor=0.1):

    assert len(y.shape) == 2
    if 0 <= smooth_factor <= 1:
        # label smoothing ref: https://www.robots.ox.ac.uk/~vgg/rg/papers/reinception.pdf
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[1]
    else:
        raise Exception(
            'Invalid label smoothing factor: ' + str(smooth_factor))
    return y


def data_flow(train_data_dir, batch_size, num_classes, input_size, preprocess_input):
    label_files = glob(os.path.join(train_data_dir, '*.txt'))
    random.shuffle(label_files)
    img_paths = []
    labels = []
    for index, file_path in enumerate(label_files):
        with codecs.open(file_path, 'r', 'utf-8') as f:
            line = f.readline( )
        line_split = line.strip( ).split(', ')
        if len(line_split) != 2:
            print('%s contain error lable' % os.path.basename(file_path))
            continue
        img_name = line_split[0]
        label = int(line_split[1])
        img_paths.append(os.path.join(train_data_dir, img_name))
        labels.append(label)
    labels = np_utils.to_categorical(labels, num_classes)
    # 标签平滑
    labels = smooth_labels(labels)
    train_img_paths, validation_img_paths, train_labels, validation_labels = \
        train_test_split(img_paths, labels, test_size=0.15, random_state=0)
    print('total samples: %d, training samples: %d, validation samples: %d' % (
        len(img_paths), len(train_img_paths), len(validation_img_paths)))

    train_sequence = BaseSequence(train_img_paths, train_labels, batch_size, [input_size, input_size],
                                  use=True, preprocess_input=preprocess_input)
    validation_sequence = BaseSequence(validation_img_paths, validation_labels, batch_size, [input_size, input_size],
                                       use=False, preprocess_input=preprocess_input)
    return train_sequence, validation_sequence


if __name__ == '__main__':
    train_data_dir = '/home/yangdd/桌面/garbage_classify_2/train_data'
    batch_size = 128
    import efficientnet.keras as efn
    # from models.inception_resnet_v2 import preprocess_input

    preprocess_input = efn.preprocess_input

    train_sequence, validation_sequence = data_flow(train_data_dir, batch_size, num_classes=40, input_size=224, preprocess_input=preprocess_input)
    for i in range(1000):
        print(i)
        batch_data, bacth_label = train_sequence.__getitem__(i)
        batch_data, bacth_label = validation_sequence.__getitem__(i)

