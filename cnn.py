def _weight_variable(shape, name=None):
    """
    Returns a variable of a specific shape.

    Parameters
    ==========
    shape : list
        shpae of variable
    name : stf
        name of variable

    Returns
    ==========
    weight : tf.Tensor
        weight variable
    """
    import tensorflow as tf
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)


def _bias_variable(shape, name=None):
    """
    Returns a variable of a specific shape.

    Parameters
    ==========
    shape : list
        shpae of variable
    name : stf
        name of variable

    Returns
    ==========
    bias : tf.Tensor
        bias variable
    """
    import tensorflow as tf
    return tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=shape), name=name)


def _conv2d(x, W, name=None):
    import tensorflow as tf
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)


def _max_pool_2x2(x, name=None):
    import tensorflow as tf
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


class Cnn:
    def __init__(self):
        import tensorflow as tf

        self.__LOAD_DATASET = False
        self.__LOAD_MODEL = False

        self._datasets = None
        self._model = None

        self._image_size = None
        self._class_num = None

        self._train_size = None
        self._test_size = None

        self._device = 'cpu'
        self._epoch = 10
        self._batch_size = 100

        self._sess = tf.Session()

    def __del__(self):
        del self.__LOAD_DATASET
        del self.__LOAD_MODEL

        del self._datasets
        del self._model

        del self._image_size
        del self._class_num

        del self._train_size
        del self._test_size

        del self._device
        del self._epoch
        del self._batch_size

        self._sess.close()
        del self._sess

    def load_dataset(self, dataset_path):
        """
        Load the dataset from a specific path.

        :param dataset_path: str
            The path to the dataset to load.
        :return: nothing
        """
        import input_data

        self.__LOAD_DATASET = True

        self._datasets = input_data.load_dataset(dataset_path)

        self._image_size = len(self._datasets.train.images[0])
        self._class_num = len(self._datasets.train.labels[0])

        self._train_size = self._datasets.train.num_examples
        self._test_size = self._datasets.test.num_examples

    def load_model(self, model_path):
        """
        Load the learned model from a specific path.

        :param model_path: str
            The path to the model to load.
        :return: nothing
        """
        import os
        import tensorflow as tf

        if os.path.exists(model_path):
            par_dir = os.path.split(model_path)[0]
            model_file = os.path.split(model_path)[1]
            if model_file.endswith('.meta'):
                if os.path.exists('{0}/checkpoint'.format(par_dir)):
                    self.__LOAD_MODEL = True
                    self._model = tf.train.import_meta_graph(model_path, clear_devices=True)
                    self._model.restore(self._sess, tf.train.latest_checkpoint(par_dir))
                else:
                    print("'checkpoint is not exist in '{}'".format(par_dir))
            else:
                print("model_path must be end with '.meta'.")
        else:
            print("The path '{}' is not Exist.".format(model_path))

    def set_device(self, device='cpu'):
        """
        Set up the device with CPU or GPU.

        :param device: str
            'cpu' or 'gpu'
        :return: nothing
        """
        if device in ['cpu', 'gpu']:
            self._device = device
        else:
            print("device must be in ['cpu', 'gpu']")

    def set_epoch(self, epoch=10):
        """
        Set size of epoch.

        :param epoch: int
            size of epoch
        :return: nothing
        """
        self._epoch = epoch

    def set_batch_size(self, batch_size=100):
        """
        Set size of batch.

        :param batch_size: int
            size of batch
        :return: nothing
        """
        self._batch_size = batch_size

    def train(self, model_save_path='./Models/model'):
        """
        Train Neural-Networks and save model.

        :param model_save_path: str
            the path to save the learning model.
        :return: nothing
        """
        if not self.__LOAD_MODEL:
            print('Please Load Dataset by load_dataset(path).')
            return

        import os
        import numpy as np
        import tensorflow as tf

        tf.set_random_seed(777)

        WIDTH = int(np.sqrt(self._image_size))
        HEIGHT = int(np.sqrt(self._image_size))
        RATIO = 1 << 3
        REDUCED_WIDTH = int(WIDTH / RATIO)
        REDUCED_HEIGHT = int(HEIGHT / RATIO)

        with tf.device('/{}:0'.format(self._device)):
            # Input
            x = tf.placeholder(tf.float32, shape=[None, self._image_size], name='input_x')
            y = tf.placeholder(tf.float32, shape=[None, self._class_num], name='input_y')

            x_image = tf.reshape(x, [-1, HEIGHT, WIDTH, 1], name='image')

            """ Weight ans Bias """
            # First Convolutional Layer
            W_conv1 = _weight_variable([5, 5, 1, 32], name='W_conv1')
            b_conv1 = _bias_variable([32], name='b_conv1')

            # Second Convolutional Layer
            W_conv2 = _weight_variable([5, 5, 32, 64], name='W_conv2')
            b_conv2 = _bias_variable([64], name='b_conv2')

            # Third Convolutional Layer
            W_conv3 = _weight_variable([5, 5, 64, 128], name='W_conv3')
            b_conv3 = _bias_variable([128], name='b_conv3')

            # Densely Connected Layer
            W_fc1 = _weight_variable([REDUCED_HEIGHT * REDUCED_WIDTH * 128, 1024], name='W_fc1')
            b_fc1 = _bias_variable([1024], name='b_fc1')

            # Readout Layer
            W_fc2 = _weight_variable([1024, self._class_num], name='W_fc2')
            b_fc2 = _bias_variable([self._class_num], name='b_fc2')

            """ Hidden Layer """
            # First Convolutional Layer
            h_conv1 = tf.nn.relu(_conv2d(x_image, W_conv1) + b_conv1, name='h_conv1')
            h_pool1 = _max_pool_2x2(h_conv1, name='h_pool1')

            # Second Convolutional Layer
            h_conv2 = tf.nn.relu(_conv2d(h_pool1, W_conv2) + b_conv2, name='h_conv2')
            h_pool2 = _max_pool_2x2(h_conv2, name='h_pool2')

            # Third Convolutional Layer
            h_conv3 = tf.nn.relu(_conv2d(h_pool2, W_conv3) + b_conv3, name='h_conv3')
            h_pool3 = _max_pool_2x2(h_conv3, name='h_pool3')

            h_pool_flat = tf.reshape(h_pool3, [-1, REDUCED_HEIGHT * REDUCED_HEIGHT * 128], name='h_pool_flat')
            h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1, name='h_fc1')

            # Dropout
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')

            y_conv = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2, name='y_conv')

            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
            train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.device('/cpu:0'):
            """
            model_param_list = [
                W_conv1, b_conv1,
                W_conv2, b_conv2,
                W_conv3, b_conv3,
                W_fc1, b_fc1,
                W_fc2, b_fc2,
                y_conv
            ]
            """
            model_saver = tf.train.Saver()

        # run session
        self._sess.run(tf.global_variables_initializer())
        self._sess.run(tf.local_variables_initializer())

        if self._train_size < self._batch_size:
            train_count = 1
        else:
            train_count = int(self._train_size / self._batch_size)

        if self._test_size < self._batch_size:
            test_count = 1
        else:
            test_count = int(self._train_size / self._batch_size)

        for _ in range(self._epoch):
            for i in range(train_count):
                batch_x, batch_y = self._datasets.train.next_batch(self._batch_size)
                self._sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.8})

        for i in range(test_count):
            batch_x, batch_y = self._datasets.test.next_batch(self._batch_size)
            self._sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})

        # save model
        par_dir = os.path.split(model_save_path)[0]
        if not os.path.exists(par_dir):
            os.makedirs(par_dir)

        model_saver.save(self._sess, model_save_path)
        self.__LOAD_MODEL = True
        print("Model save to '{}.meta'".format(model_save_path))

    def query(self, images, model_path=None):
        """
        Input images and predict labels.

        :param images: array
        :param model_path: str
        :return: nothing
        """
        if not model_path:
            if not self.__LOAD_MODEL:
                print('Please Load Model by load_model(path).')
                return
        else:
            self.load_model(model_path)

        import tensorflow as tf

        graph = tf.get_default_graph()

        with tf.device('/{}:0'.format(self._device)):
            # Input
            x = graph.get_tensor_by_name('input_x:0')

            # Dropout
            keep_prob = graph.get_tensor_by_name('keep_prob:0')

            y_conv = graph.get_tensor_by_name('y_conv:0')
            predict = tf.argmax(y_conv, 1)

        # run session
        result = self._sess.run(predict, feed_dict={x: images, keep_prob: 1.0})

        return result
