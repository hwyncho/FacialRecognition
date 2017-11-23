"""
Created by 조휘연 on 2017. 08. 24.
Last updated by 조휘연 on 2017. 11. 23.
Copyright © 2017년 조휘연. All rights reserved.
==================================================
Convert Images to Dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import struct
import sys

from PIL import Image
import numpy as np


random.seed(777)


IMAGES_DIR = './Images'

IMAGE_WIDTH = 48
IMAGE_HEIGHT = 48
IMAGE_MODE = 'RGB'
#IMAGE_MODE = 'L'


def _is_image(image_name):
    if not image_name.startswith('.'):
        if image_name.endswith('.jpg') or image_name.endswith('.jpeg') or image_name.endswith('.png'):
            return True

    return False


def _sampling_over():
    label_path_list = list()
    count_list = list()

    # count the number of images per label
    for label_name in os.listdir(IMAGES_DIR):
        label_path = '{0}/{1}'.format(IMAGES_DIR, label_name)
        if (not label_name.startswith('.')) and os.path.isdir(label_path):
            count = len(os.listdir(label_path))
            if count > 0:
                label_path_list.append(label_path)
                count_list.append(count)

    image_path_list = list()
    max_count = max(count_list)
    max_idx = np.argmax(count_list)

    for (idx, label_path) in enumerate(label_path_list):
        image_name_list = list()
        for image_name in os.listdir(label_path):
            if _is_image(image_name):
                image_name_list.append(image_name)

        if idx != max_idx:
            random.shuffle(image_name_list)
            a = max_count // len(image_name_list)
            b = max_count % len(image_name_list)
            image_name_list = image_name_list * a + image_name_list[:b]

        for image_name in image_name_list:
            image_path = '{0}/{1}'.format(label_path, image_name)
            image_path_list.append(image_path)

    return label_path_list, image_path_list


def _sampling_under():
    label_path_list = list()
    count_list = list()

    # count the number of images per label
    for label_name in os.listdir(IMAGES_DIR):
        label_path = '{0}/{1}'.format(IMAGES_DIR, label_name)
        if (not label_name.startswith('.')) and os.path.isdir(label_path):
            count = len(os.listdir(label_path))
            if count > 0:
                label_path_list.append(label_path)
                count_list.append(count)

    image_path_list = list()
    min_count = min(count_list)
    min_idx = np.argmin(count_list)

    for (idx, label_path) in enumerate(label_path_list):
        image_name_list = list()
        for image_name in os.listdir(label_path):
            if _is_image(image_name):
                image_name_list.append(image_name)

        if idx != min_idx:
            random.shuffle(image_name_list)
            image_name_list = image_name_list[:min_count]

        for image_name in image_name_list:
            image_path = '{0}/{1}'.format(label_path, image_name)
            image_path_list.append(image_path)

    return label_path_list, image_path_list


def _make_list(sampling=None):
    if sampling:
        if sampling.lower() == 'over':
            return _sampling_over()
        elif sampling.lower() == 'under':
            return _sampling_under()
    else:
        label_path_list = list()

        for label_name in os.listdir(IMAGES_DIR):
            label_path = '{0}/{1}'.format(IMAGES_DIR, label_name)
            if (not label_name.startswith('.')) and os.path.isdir(label_path):
                if len(os.listdir(label_path)) > 0:
                    label_path_list.append(label_path)

        image_path_list = list()

        for label_path in label_path_list:
            for image_name in os.listdir(label_path):
                if _is_image(image_name):
                    image_path = '{0}/{1}'.format(label_path, image_name)
                    image_path_list.append(image_path)

        return label_path_list, image_path_list


def main(save_path='./Datasets.bin', shuffle=True, sampling=None):
    """
    Parameters
    ===========
    save_path : str
        path to save converted dataset
    shuffle : bool
        whether to shuffle image list
    sampling : str
        sampling mode
    """
    if not isinstance(save_path, str):
        raise TypeError("type of 'save_path' must be 'str'.")

    if not isinstance(shuffle, bool):
        raise TypeError("type of 'shuffle' must be 'bool'.")

    if sampling:
        if not isinstance(sampling, str):
            raise TypeError("type of 'sampling' must be 'str'.")
        else:
            if (sampling.lower() != 'over') or (sampling.lower() != 'under'):
                raise TypeError("'sampling' must be 'over' or 'under'.")

    label_path_list, image_path_list = _make_list(sampling)

    label_dict = dict()
    for (idx, label_path) in enumerate(label_path_list):
        label_dict[label_path] = idx

    if shuffle:
        random.shuffle(image_path_list)

    # convert images to dataset
    f = open(save_path, mode='wb')

    if IMAGE_MODE == 'RGB':
        info = [(IMAGE_HEIGHT * IMAGE_WIDTH * 3), len(label_path_list), len(image_path_list)]
    elif IMAGE_MODE == 'L':
        info = [(IMAGE_HEIGHT * IMAGE_WIDTH), len(label_path_list), len(image_path_list)]
    else:
        raise ValueError("'IMAGE_MODE' must be 'RGB' or 'L'.")

    # write dataset information
    # image_size, label_size, num_examples
    buffer = struct.pack('%si' % len(info), *info)
    f.write(buffer)

    for image_path in image_path_list:
        image = Image.open(image_path).convert(mode=IMAGE_MODE)
        w, h = image.size

        if (w * h) != (IMAGE_HEIGHT * IMAGE_WIDTH):
            raise ValueError(
                "size of image must be '{0}', but '{1}' is '{2}'.".format(IMAGE_HEIGHT * IMAGE_WIDTH, image, (w * h))
            )

        pixel_list = list()
        for y in range(h):
            for x in range(w):
                pixel = image.getpixel((x, y))
                if isinstance(pixel, tuple):
                    pixel_list += [p for p in pixel]
                else:
                    pixel_list.append(pixel)

        label_path = os.path.split(image_path)[0]
        label = label_dict[label_path]

        line = pixel_list + [label]
        buffer = struct.pack('%si' % len(line), *line)
        f.write(buffer)

    f.close()


if __name__ == '__main__':
    if len(sys.argv) > 3:
        raise RuntimeError('python convert_images.py [images_dir] [save_path]')

    if len(sys.argv) == 3:
        IMAGES_DIR = sys.argv[1]
        main(save_path=sys.argv[2])
