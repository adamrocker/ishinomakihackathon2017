# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
# sys.path.insert(0, '/usr/local/lib/python2.7/site-packages')
sys.path.append('/lib/python2.7/site-packages')


import numpy as np
import tensorflow as tf
import random

N_DIGITS = 10  # Number of digits.
X_FEATURE = 'x'  # Name of the input feature.


def conv_model(features, labels, mode):
    """2-layer convolution model."""
    # Reshape feature to 4d tensor with 2nd and 3rd dimensions being
    # image width and height final dimension being the number of color channels.
    feature = tf.reshape(features[X_FEATURE], [-1, 28, 28, 1])

    # First conv layer will compute 32 features for each 5x5 patch
    with tf.variable_scope('conv_layer1'):
        h_conv1 = tf.layers.conv2d(
            feature,
            filters=32,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu)
        h_pool1 = tf.layers.max_pooling2d(
            h_conv1, pool_size=2, strides=2, padding='same')

    # Second conv layer will compute 64 features for each 5x5 patch.
    with tf.variable_scope('conv_layer2'):
        h_conv2 = tf.layers.conv2d(
            h_pool1,
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu)
        h_pool2 = tf.layers.max_pooling2d(
            h_conv2, pool_size=2, strides=2, padding='same')
        # reshape tensor into a batch of vectors
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    # Densely connected layer with 1024 neurons.
    h_fc1 = tf.layers.dense(h_pool2_flat, 1024, activation=tf.nn.relu)
    if mode == tf.estimator.ModeKeys.TRAIN:
        h_fc1 = tf.layers.dropout(h_fc1, rate=0.5)

    # Compute logits (1 per class) and compute loss.
    logits = tf.layers.dense(h_fc1, N_DIGITS, activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class': predicted_classes,
            'prob': tf.nn.softmax(logits)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    onehot_labels = tf.one_hot(tf.cast(labels, tf.int32), N_DIGITS, 1, 0)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # Create training op.
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    # Compute evaluation metrics.
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predicted_classes)
    }
    return tf.estimator.EstimatorSpec(
        mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main0(unused_args):
    ### Download and load MNIST dataset.
    mnist = tf.contrib.learn.datasets.DATASETS['mnist']('/tmp/mnist')
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={X_FEATURE: mnist.train.images},
        y=mnist.train.labels.astype(np.int32),
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={X_FEATURE: mnist.train.images},
        y=mnist.train.labels.astype(np.int32),
        num_epochs=1,
        shuffle=False)

    ### Linear classifier.
    feature_columns = [
        tf.feature_column.numeric_column(
            X_FEATURE, shape=mnist.train.images.shape[1:])]
    classifier = tf.estimator.LinearClassifier(
        feature_columns=feature_columns, n_classes=N_DIGITS)
    classifier.train(input_fn=train_input_fn, steps=200)
    scores = classifier.evaluate(input_fn=test_input_fn)
    print('Accuracy (LinearClassifier): {0:f}'.format(scores['accuracy']))

    ### Convolutional network
    classifier = tf.estimator.Estimator(model_fn=conv_model)
    classifier.train(input_fn=train_input_fn, steps=200)
    scores = classifier.evaluate(input_fn=test_input_fn)
    print('Accuracy (conv_model): {0:f}'.format(scores['accuracy']))


def main1(unused_args):
    ### Download and load MNIST dataset.
    mnist = tf.contrib.learn.datasets.DATASETS['mnist']('/tmp/mnist')
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={X_FEATURE: mnist.train.images},
        y=mnist.train.labels.astype(np.int32),
        num_epochs=1,
        shuffle=False)

    ### Convolutional network
    classifier = tf.estimator.Estimator(model_fn=conv_model)
    classifier.train(input_fn=test_input_fn, steps=1)
    scores = classifier.evaluate(input_fn=test_input_fn)
    print('Accuracy (conv_model): {0:f}'.format(scores['accuracy']))
    print(scores)


def main2(unused_args):
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={X_FEATURE: mnist.train.images},
        y=mnist.train.labels.astype(np.int32),
        num_epochs=1,
        shuffle=False)

    ### Linear classifier.
    feature_columns = [
        tf.feature_column.numeric_column(
            X_FEATURE, shape=mnist.train.images.shape[1:])]
    classifier = tf.estimator.LinearClassifier(
        feature_columns=feature_columns, n_classes=N_DIGITS)
    classifier.train(input_fn=test_input_fn, steps=0)
    scores = classifier.evaluate(input_fn=test_input_fn)
    print('Accuracy (LinearClassifier): {0:f}'.format(scores['accuracy']))


def main3():
    u"""
    センサー入力
    食べ物との距離 7=(0, 30, 60, 90, 120, 150,180deg), 物体との距離7=[0, 30, 60, 90, 150, 180deg]
    """
    NUM_FEATURE = 4
    NUM_HIDDEN1 = 2
    NUM_OUTPUT = 2
    x = tf.placeholder(tf.float32, [None, NUM_FEATURE])
    z1 = tf.zeros([NUM_FEATURE, NUM_HIDDEN1])
    w1 = tf.Variable(z1)
    b1 = tf.Variable(tf.zeros([NUM_HIDDEN1]))
    x1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
    y_ = tf.placeholder(tf.float32, [None, NUM_OUTPUT])
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(x1), reduction_indices=[1]))
    # train_step = tf.train.GradientDescentOptimizer(0.0).minimize(cross_entropy)
    with tf.Session():
        tf.global_variables_initializer().run()
        for _ in range(10):
            batch_xs = np.array([[i for i in range(NUM_FEATURE)]])
            batch_ys = np.array([[i for i in range(NUM_OUTPUT)]])
            fetch = x1.eval(feed_dict={x: batch_xs, y_: batch_ys})
            print(fetch)


class Genome(object):

    NUM_FEATURE = 3
    NUM_HIDDEN_NODE = [2]  # 各レイヤーのノード数
    NUM_HIDDEN = len(NUM_HIDDEN_NODE)  # レイヤー数
    NUM_OUTPUT = 2
    ARRAY = [NUM_FEATURE] + NUM_HIDDEN_NODE + [NUM_OUTPUT]
    LAYER = len(ARRAY) - 1
    GENE_LENGTH = -1
    MUTATION_RATE = 0.01

    @classmethod
    def gene_length(cls):
        if 0 <= cls.GENE_LENGTH:
            return cls.GENE_LENGTH

        length = 0
        for index, i in enumerate(cls.ARRAY[0: -1]):
            length += i * cls.ARRAY[index + 1]
        cls.GENE_LENGTH = length
        return cls.GENE_LENGTH

    def __init__(self, mutation_rate=None):
        length = self.gene_length()
        self._gene = np.random.rand(length).astype(np.float32) - np.random.rand(length).astype(np.float32)
        if mutation_rate:
            self._mutation_rate = mutation_rate
        else:
            self._mutation_rate = 1.0 / length
        self._fitness = 0.0

    def _gene_layer_offset(self, layer):
        # layerのスタート地点までのオフセット
        length = 0
        for index, i in enumerate(Genome.ARRAY[0: layer]):
            length += i * Genome.ARRAY[index + 1]
        return length

    def gene_layer(self, layer):
        start = self._gene_layer_offset(layer)
        end = self._gene_layer_offset(layer + 1)
        return self._gene[start:end]

    def mutate(self):
        length = len(self._gene)
        mutate = np.zeros(length).astype(np.float32)
        rand = np.random.rand(length)
        for i in range(length):
            if rand[i] <= self._mutation_rate:
                val = np.random.rand() - np.random.rand()
                print("mutate[%d] = %f" % (i, val))
                mutate[i] = val
            else:
                mutate[i] = 0.0

        self._gene += mutate

    def set_fitness(self, fitness):
        self._fitness = fitness


class Population(object):

    def __init__(self, size):
        self._generation = 0  # 世代数
        self._size = size
        self._population = []
        for i in range(size):
            self._population.append(Genome())

    def get_genome(self, index):
        # type: (int) -> Genome
        return self._population[index]

    def flatten(self):
        size = self._size
        gene_length = Genome.gene_length()
        index = 0
        flat = np.zeros(size * gene_length)
        for layer in range(Genome.LAYER):
            for genome in self._population:
                arr = genome.gene_layer(layer)
                end = index + len(arr)
                flat[index:end] = arr
                index = end

        return flat



def main(args):
    """
    センサー入力
    食べ物との距離 7=(0, 30, 60, 90, 120, 150,180deg), 物体との距離7=[0, 30, 60, 90, 150, 180deg]
    """
    POPULATION = 2
    NUM_FEATURE = 3
    NUM_HIDDEN1 = 2
    NUM_OUTPUT = 2
    x = tf.placeholder(tf.float32, [POPULATION, None, NUM_FEATURE], name="input")
    pop = [1, 2, 3, 4, 5, 6, 2, 4, 6, 8, 10, 12]  # 前半6つが1個体の染色体。後半6つが2個体目。
    c1 = tf.constant(pop,
                     dtype=tf.float32, shape=[POPULATION, NUM_FEATURE, NUM_HIDDEN1],
                     name="layer1")
    w1 = tf.Variable(c1)
    b1 = tf.Variable(tf.zeros([POPULATION, NUM_HIDDEN1]))
    # x1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
    x1 = tf.matmul(x, w1)
    """
    c2 = tf.constant([i for i in range(NUM_HIDDEN1 * NUM_OUTPUT)],
                     dtype=tf.float32, shape=[NUM_HIDDEN1, NUM_OUTPUT],
                     name="layer2")
    w2 = tf.Variable(c2)
    x2 = tf.nn.sigmoid(tf.matmul(x1, w2))
    """
    sess = tf.Session()
    with sess.as_default():
        tf.global_variables_initializer().run()

        for _ in range(1):
            batch_xs = np.array([[[1 for _ in range(NUM_FEATURE)]] for _ in range(POPULATION)])
            print (batch_xs)
            fetch = x1.eval(feed_dict={x: batch_xs})
            print(fetch)


if __name__ == '__main__':
    # tf.app.run()
    p = Population(2)
    print(p.flatten())
    print(p.get_genome(0)._gene)
