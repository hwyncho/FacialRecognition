"""
Created by 조휘연 on 2017. 08. 24.
Last updated by 조휘연 on 2017. 11. 24.
Copyright © 2017년 조휘연. All rights reserved.
==================================================
Convolutional Neural Network.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
from PIL import Image

from dataset import Dataset

np.random.seed(777)
tf.set_random_seed(777)


class Cnn:
    def __init__(self):
        self.__load_dataset = False
        self.__load_model = False

        self._dataset = None
        self._model = None

        self._image_width = 48
        self._image_height = 48
        self._image_mode = 'RGB'

        self._dataset_size = 0
        self._image_size = 0
        self._label_size = 0

        self._device = 'cpu'
        self._epoch = 100
        self._batch_size = 100

    def __del__(self):
        del self.__load_dataset
        del self.__load_model

        del self._dataset
        del self._model

        del self._image_width
        del self._image_height
        del self._image_mode

        del self._dataset_size
        del self._image_size
        del self._label_size

        del self._device
        del self._epoch
        del self._batch_size

    def load_dataset(self, path):
        """
        Load the dataset from a specific path.

        :param dataset_path: str
            The path to the dataset to load.
        """
        if not isinstance(path, str):
            raise TypeError("type of 'path' must be str.")

        if not os.path.exists(path):
            raise FileNotFoundError()

        self.__load_dataset = True
        self._dataset = Dataset(path, one_hot=True)
        self._dataset_size = self._dataset.num_examples
        self._image_size = self._dataset.image_size
        self._label_size = self._dataset.label_size

    def load_model(self, path, sess):
        """
        Load the learned model from a specific path.

        :param model_path: str
            The path to the model to load.
        """
        if not isinstance(path, str):
            raise TypeError("type of 'path' must be str.")

        par_dir = os.path.split(path)[0]
        meta_file = '{}.meta'.format(path)
        ckpt_file = '{}/checkpoint'.format(par_dir)

        if not os.path.exists(meta_file):
            raise FileNotFoundError('The meta file is not exist from: {}'.format(par_dir))

        if not os.path.exists(ckpt_file):
            raise FileNotFoundError('The checkpoint file is not exist from: {}'.format(par_dir))

        self.__load_model = True
        self._model = tf.train.import_meta_graph(meta_file, clear_devices=True)
        self._model.restore(sess, tf.train.latest_checkpoint(par_dir))

    def set_device(self, device='cpu'):
        """
        Set up the device with CPU or GPU.

        :param device: str
            'cpu' or 'gpu'
        """
        if device in ['cpu', 'gpu']:
            self._device = device
        else:
            raise ValueError("'device' must be in 'cpu' or 'gpu'.")

    def set_epoch(self, epoch=100):
        """
        Set size of epoch.

        :param epoch: int
            size of epoch
        """
        self._epoch = epoch

    def set_batch_size(self, batch_size=100):
        """
        Set size of batch.

        :param batch_size: int
            size of batch
        """
        self._batch_size = batch_size

    def train(self, dataset_path, model_path='./Models/model'):
        """
        Train Neural-Networks and save model.

        :param dataset_path: str
            the path to load the dataset for train.
        :param model_path: str
            the path to save the learning model.
        """
        if not isinstance(dataset_path, str):
            raise TypeError("type of 'dataset_path' must be str.")

        if not isinstance(model_path, str):
            raise TypeError("type of 'model_path' must be str.")

        self.load_dataset(dataset_path)

        image_width = 48
        image_height = 48
        if self._image_mode == 'RGB':
            image_channel = 3
        elif self._image_mode == 'L':
            image_channel = 1

        learning_rate = 0.001

        with tf.device('/{}:0'.format(self._device)):
            # Input
            with tf.variable_scope('Input_Layer') as scope:
                x = tf.placeholder(tf.float32, shape=[None, self._image_size], name='images')
                y = tf.placeholder(tf.float32, shape=[None, self._label_size], name='labels')

            x_2d = tf.reshape(x, [-1, image_height, image_width, image_channel], name='images_2d')

            # Hidden Layers
            with tf.variable_scope('Hidden_Layers') as scope:
                conv_0 = tf.layers.conv2d(
                    inputs=x_2d, filters=32, kernel_size=5, padding='same', activation=tf.nn.relu
                )
                pool_0 = tf.layers.max_pooling2d(
                    inputs=conv_0, pool_size=2, strides=2, padding='same'
                )

                conv_1 = tf.layers.conv2d(
                    inputs=pool_0, filters=64, kernel_size=5, padding='same', activation=tf.nn.relu
                )
                pool_1 = tf.layers.max_pooling2d(
                    inputs=conv_1, pool_size=2, strides=2, padding='same'
                )

                conv_2 = tf.layers.conv2d(
                    inputs=pool_1, filters=128, kernel_size=5, padding='same', activation=tf.nn.relu
                )
                pool_2 = tf.layers.max_pooling2d(
                    inputs=conv_2, pool_size=2, strides=2, padding='same'
                )

                conv_3 = tf.layers.conv2d(
                    inputs=pool_2, filters=256, kernel_size=5, padding='same', activation=tf.nn.relu
                )
                pool_3 = tf.layers.max_pooling2d(
                    inputs=conv_3, pool_size=2, strides=2, padding='same'
                )

                flatten = tf.layers.flatten(inputs=pool_3)
                fc = tf.layers.dense(inputs=flatten, units=1024, activation=tf.nn.relu)

            # Output
            with tf.variable_scope('Output_Layer') as scope:
                y_predict = tf.layers.dense(inputs=fc, units=self._label_size, activation=None, name='y_predict')

            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_predict))
            train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

        with tf.device('/cpu:0'):
            model_saver = tf.train.Saver()

        # set train count
        if self._dataset_size <= self._batch_size:
            train_count = 1
        else:
            if (self._dataset_size % self._batch_size) == 0:
                train_count = int(self._dataset_size / self._batch_size)
            else:
                train_count = int(self._dataset_size / self._batch_size) + 1

        # session configure
        config = tf.ConfigProto(log_device_placement=True)
        if self._device == 'gpu':
            config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # optimization
            for step in range(self._epoch):
                self._dataset.reset_batch()
                for i in range(train_count):
                    batch_x, batch_y = self._dataset.next_batch(self._batch_size)
                    sess.run(train_step, feed_dict={x: batch_x, y: batch_y})

            # save model
            par_dir = os.path.split(model_path)[0]
            if not os.path.exists(par_dir):
                os.makedirs(par_dir)

            model_saver.save(sess, model_path)
            self.__load_model = True
            print("Model save to '{}'".format(model_path))

            tf.reset_default_graph()

    def eval(self, dataset_path, model_path):
        """
        Evaluate saved model.

        :param dataset_path: str
            the path to load the dataset for evaluate.
        :param model_path: str
            the path to load the learned model.

        :return: dict
        """
        if not isinstance(dataset_path, str):
            raise TypeError("type of 'dataset_path' must be str.")

        if not isinstance(model_path, str):
            raise TypeError("type of 'model_path' must be str.")

        self.load_dataset(dataset_path)

        sess = tf.Session()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        self.load_model(model_path, sess)
        graph = tf.get_default_graph()

        with tf.device('/{}:0'.format(self._device)):
            # Input
            x = graph.get_tensor_by_name('Input_Layer/input_x:0')
            y = graph.get_tensor_by_name('Input_Layer/input_y:0')

            y_predict = graph.get_tensor_by_name('Output_Layer/y_predict/BiasAdd:0')

            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_predict, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.device('/cpu:0'):
            confusion_matrix = tf.confusion_matrix(labels=tf.argmax(y, 1), predictions=tf.argmax(y_predict, 1))

        if self._dataset_size <= self._batch_size:
            test_count = 1
        else:
            if (self._dataset_size % self._batch_size) == 0:
                test_count = int(self._dataset_size / self._batch_size)
            else:
                test_count = int(self._dataset_size / self._batch_size) + 1

        # test
        self._dataset.reset_batch()
        for i in range(test_count):
            batch_x, batch_y = self._dataset.next_batch(self._batch_size)
            if i == 0:
                test_accuracy = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                test_cm = sess.run(confusion_matrix, feed_dict={x: batch_x, y: batch_y})
            else:
                test_accuracy += sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                test_cm += sess.run(confusion_matrix, feed_dict={x: batch_x, y: batch_y})

        test_accuracy /= test_count
        test_cm = test_cm.tolist()

        tf.reset_default_graph()
        sess.close()

        return {
            'accuracy': test_accuracy,
            'confusion_matrix': test_cm
        }

    def query(self, image_path, model_path):
        """
        Input images and predict labels.

        :param images: array
        :param model_path: str

        :return: nothing
        """
        if not isinstance(image_path, str):
            raise TypeError("type of 'image_path' must be str.")

        if not isinstance(model_path, str):
            raise TypeError("type of 'model_path' must be str.")

        image_width = 48
        image_height = 48
        if self._image_mode == 'RGB':
            image_channel = 3
        elif self._image_mode == 'L':
            image_channel = 1

        image = Image.open(image_path).convert(mode=self._image_mode).resize((image_width, image_height))
        w, h = image.size

        pixel_list = list()
        for y in range(h):
            for x in range(w):
                pixel = image.getpixel((x, y))
                if isinstance(pixel, tuple):
                    pixel_list += [p for p in pixel]
                else:
                    pixel_list.append(pixel)

        sess = tf.Session()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        self.load_model(model_path, sess)
        graph = tf.get_default_graph()

        with tf.device('/{}:0'.format(self._device)):
            # Input
            x = graph.get_tensor_by_name('Input_Layer/input_x:0')

            y_predict = graph.get_tensor_by_name('Output_Layer/y_predict/BiasAdd:0')

            predict = tf.argmax(y_predict, 1)

            # run session
            result = sess.run(predict, feed_dict={x: [pixel_list]})

        sess.close()

        return result
