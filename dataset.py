"""
Created by hwyncho on 2017. 08. 24.
Last updated by hwyncho on 2018. 06. 16.
Copyright © 2018 hwyncho. All rights reserved.
==================================================
Class of Dataset
"""
import os
import struct


class Dataset:
    """ Dataset class """

    def __init__(self, path, one_hot=True):
        """ Dataset constructor """
        self._fp = None
        self._one_hot = one_hot
        self.image_size = 0
        self.label_size = 0
        self.num_examples = 0
        self.images = list()
        self.labels = list()
        self._load_dataset(path, one_hot)

    def __del__(self):
        """ Dataset deconstructor """
        self._fp.close()
        del self._fp
        del self._one_hot
        del self.image_size
        del self.label_size
        del self.num_examples
        del self.images
        del self.labels

    def _load_dataset(self, path, one_hot=True):
        if not isinstance(path, str):
            raise TypeError("type of 'path' must be str.")

        if not os.path.exists(path):
            raise FileNotFoundError(
                'The dataset file is not exist from: {}'.format(path))

        self._fp = open(path, mode='rb')

        buffer = self._fp.read(12)
        info = struct.unpack('3i', buffer)
        self.image_size = info[0]
        self.label_size = info[1]
        self.num_examples = info[2]

    def reset_batch(self):
        """ Reset index of batch. """
        self._fp.seek(12)

    def next_batch(self, batch_size):
        """
        Returns the next batch.

        :param batch_size: int
            size of batch

        :return: tuple
            returned batch
        """
        self.images.clear()
        self.labels.clear()

        for _ in range(batch_size):
            buffer = self._fp.read((self.image_size + 1) * 4)
            if not buffer:
                self.reset_batch()
                break

            line = struct.unpack('%si' % (self.image_size + 1), buffer)

            # image
            pixel_list = line[:self.image_size]
            self.images.append(pixel_list)

            # label
            if self._one_hot:
                label = [0 for _ in range(self.label_size)]
                label[line[-1]] = 1
            else:
                label = line[-1]
            self.labels.append(label)

        return self.images.copy(), self.labels.copy()
