def _weight_variable(shape):
    import tensorflow as tf

    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def _bias_variable(shape):
    import tensorflow as tf

    return tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=shape))


def _conv2d(x, W):
    import tensorflow as tf

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def _max_pool_2x2(x):
    import tensorflow as tf

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

class Cnn:
    def __init__(self):
        self._datasets = None
        self._train_size = None
        self._test_size = None
        self._image_size = None
        self._class_num = None
        self._LOAD_FLAG = False
        self._model = None
        self._epoch = 10
        self._batch_size = 100
        self._device = 'cpu'

    def load_dataset(self, dataset_path='./Datasets.zip'):
        import input_data

        self._datasets = input_data.load_dataset(dataset_path)
        self._train_size = len(self._datasets.train.images)
        self._test_size = len(self._datasets.test.images)
        self._image_size = len(self._datasets.train.images[0])
        self._class_num = len(self._datasets.train.labels[0])
        self._LOAD_FLAG = True

    def load_model(self, model_path='./Model.json'):
        import json
        import os

        if os.path.exists(model_path):
            with open(model_path, 'r', encoding='utf-8') as f:
                self._model = json.load(f)
        else:
            self._model = None

    def set_epoch(self, epoch=10):
        self._epoch = epoch

    def set_batch_size(self, batch_size=100):
        self._batch_size = batch_size

    def set_device(self, device='cpu'):
        if device in ['cpu', 'gpu']:
            self._device = device
        else:
            print("device must be in ['cpu', 'gpu']")

    def train(self):
        if not self._LOAD_FLAG:
            print('Please Load Dataset by load_dataset(path).')
            return

        import json
        import numpy as np
        import tensorflow as tf

        with tf.device('/{}:0'.format(self._device)):
            # Input
            x = tf.placeholder(tf.float32, shape=[None, self._image_size], name='float')
            y = tf.placeholder(tf.float32, shape=[None, self._class_num], name='float')

            width = int(np.sqrt(self._image_size))
            height = int(np.sqrt(self._image_size))

            x_image = tf.reshape(x, [-1, height, width, 1])

            # First Convolutional Layer
            W_conv1 = _weight_variable([5, 5, 1, 32])
            b_conv1 = _bias_variable([32])

            h_conv1 = tf.nn.relu(_conv2d(x_image, W_conv1) + b_conv1)
            h_pool1 = _max_pool_2x2(h_conv1)

            # Second Convolutional Layer
            W_conv2 = _weight_variable([5, 5, 32, 64])
            b_conv2 = _bias_variable([64])

            h_conv2 = tf.nn.relu(_conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = _max_pool_2x2(h_conv2)

            # Third Convolutional Layer
            W_conv3 = _weight_variable([5, 5, 64, 128])
            b_conv3 = _bias_variable([128])

            h_conv3 = tf.nn.relu(_conv2d(h_pool2, W_conv3) + b_conv3)
            h_pool3 = _max_pool_2x2(h_conv3)

            # Densely Connected Layer
            a = 1 << 3
            W_fc1 = _weight_variable([int(height / a) * int(width / a) * 128, 1024])
            b_fc1 = _bias_variable([1024])

            h_pool_flat = tf.reshape(h_pool3, [-1, int(height / a) * int(width / a) * 128])
            h_fc1 = tf.nn.tanh(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

            # Dropout
            keep_prob = tf.placeholder(tf.float32, name='float')
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            # Readout Layer
            W_fc2 = _weight_variable([1024, self._class_num])
            b_fc2 = _bias_variable([self._class_num])

            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
            train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            train_count = int(self._train_size / self._batch_size)
            test_count = int(self._train_size / self._batch_size)

            for _ in range(self._epoch):
                for i in range(train_count):
                    batch_x, batch_y = self._datasets.train.next_batch(self._batch_size)
                    sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.8})

            for i in range(test_count):
                batch_x, batch_y = self._datasets.test.next_batch(self._batch_size)
                sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})

        self._model = dict()
        self._model['W_conv1'] = W_conv1
        self._model['b_conv1'] = b_conv1
        self._model['W_conv2'] = W_conv2
        self._model['b_conv2'] = b_conv2
        self._model['W_conv3'] = W_conv3
        self._model['b_conv3'] = b_conv3
        self._model['W_fc1'] = W_fc1
        self._model['b_fc1'] = b_fc1
        self._model['W_fc2'] = W_fc2
        self._model['b_fc2'] = b_fc2

        with open('./Model.json', 'w', encoding='utf-8') as f:
            json.dump(self._model, f)

    def query(self, image, model_path=None):
        if not model_path:
            if not self._model:
                print('Please Load Model by load_model(path).')
                return
        else:
            self.load_model(model_path)

        import numpy as np
        import tensorflow as tf

        with tf.device('/{}:0'.format(self._device)):
            # Input
            x = tf.placeholder(tf.float32, shape=[self._image_size], name='float')

            width = int(np.sqrt(self._image_size))
            height = int(np.sqrt(self._image_size))

            x_image = tf.reshape(x, [-1, height, width, 1])

            # First Convolutional Layer
            W_conv1 = self._model['W_conv1']
            b_conv1 = self._model['b_conv1']

            h_conv1 = tf.nn.relu(self._conv2d(x_image, W_conv1) + b_conv1)
            h_pool1 = self._max_pool_2x2(h_conv1)

            # Second Convolutional Layer
            W_conv2 = self._model['W_conv2']
            b_conv2 = self._model['b_conv2']

            h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = self._max_pool_2x2(h_conv2)

            # Third Convolutional Layer
            W_conv3 = self._model['W_conv3']
            b_conv3 = self._model['b_conv3']

            h_conv3 = tf.nn.relu(self._conv2d(h_pool2, W_conv3) + b_conv3)
            h_pool3 = self._max_pool_2x2(h_conv3)

            # Densely Connected Layer
            W_fc1 = self._model['W_fc1']
            b_fc1 = self._model['b_fc']

            h_pool_flat = tf.reshape(h_pool3, [-1, 6 * 6 * 128])
            h_fc1 = tf.nn.tanh(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

            # Readout Layer
            W_fc2 = self._model['W_fc2']
            b_fc2 = self._model['b_fc2']

            y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
            predict = tf.argmax(y_conv, 1)

        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            result = sess.run(predict, feed_dict={x: image})

            return result
