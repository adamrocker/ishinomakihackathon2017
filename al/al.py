# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
sys.path.append('/lib/python2.7/site-packages')


import numpy as np
import tensorflow as tf


def print_debug(msg):
    if False:
        print(msg)


class Genome(object):

    NUM_FEATURE = 6  # [壁距離, 壁角度, 餌距離, 餌角度, 敵距離, 敵距離]  # それぞれ1つしか認識できない
    NUM_HIDDEN_NODE = [4]  # 各レイヤーのノード数
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
                print_debug("mutate[%d] = %f" % (i, val))
                mutate[i] = val
            else:
                mutate[i] = 0.0

        self._gene += mutate

    def copy_gene(self):
        # deep copy
        return self._gene.copy()

    def set_gene(self, gene):
        self._gene = gene

    def set_fitness(self, fitness):
        self._fitness = fitness

    def get_fitness(self):
        return self._fitness


class NN(object):

    def __init__(self, population):
        self._x, self._model = self._build_nn(population)

    @classmethod
    def _flatten(cls, population):
        # make flatten all genome info for GPU calculation
        size = len(population)
        gene_length = Genome.gene_length()
        index = 0
        flat = np.zeros(size * gene_length)
        for layer in range(Genome.LAYER):
            for genome in population:
                arr = genome.gene_layer(layer)
                end = index + len(arr)
                flat[index:end] = arr
                index = end

        return flat

    @classmethod
    def _build_nn(cls, population):
        size = len(population)
        x = tf.placeholder(tf.float32, [size, None, Genome.NUM_FEATURE], name="input")
        genes = cls._flatten(population)
        print_debug("----genes-----")
        print_debug(genes)

        start = 0
        length = size * Genome.NUM_FEATURE * Genome.NUM_HIDDEN_NODE[0]
        c1 = tf.constant(genes[start:length],
                         dtype=tf.float32,
                         shape=[size, Genome.NUM_FEATURE, Genome.NUM_HIDDEN_NODE[0]],
                         name="layer1")
        w1 = tf.Variable(c1)
        x1 = 2 * tf.nn.sigmoid(tf.matmul(x, w1)) - 1
        print_debug("----layer1[{}:{}]----".format(start, length))
        print_debug(genes[start:length])

        start = length
        length = start + size * Genome.NUM_HIDDEN_NODE[0] * Genome.NUM_OUTPUT
        c2 = tf.constant(genes[start:length],
                         dtype=tf.float32,
                         shape=[size, Genome.NUM_HIDDEN_NODE[0], Genome.NUM_OUTPUT],
                         name="layer2")
        w2 = tf.Variable(c2)
        x2 = tf.nn.sigmoid(tf.matmul(x1, w2))
        print_debug("----layer2[{}:{}]----".format(start, length))
        print_debug(genes[start:length])
        return x, x2

    def eval(self, input):
        fetch = self._model.eval(feed_dict={self._x: input})
        return fetch[:, 0]


class GenePool(object):

    def __init__(self, size):
        self._generation = 0  # 世代数
        self._selection_size = 2
        self._size = size
        self._population = []
        for i in range(size):
            self._population.append(Genome())

        self._nn = None

    def get_genome(self, index):
        # type: (int) -> Genome
        return self._population[index]

    def print_all_genome(self):
        for genome in self._population:
            print(genome._gene)

    def init_world(self):
        self._nn = NN(self._population)

    def play(self, input):
        return self._nn.eval(input=input)

    def mutation(self):
        for genome in self._population:
            genome.mutate()

    def selection(self):
        self._tournament_selection()

    def _tournament_selection(self):
        index_arr = np.random.randint(0, self._size, self._selection_size)
        winner = self._population[0]  # type: Genome
        losers = []  # List[Genome]
        for i in range(1, self._selection_size):
            winner_fitness = winner.get_fitness()
            challenger_index = index_arr[i]
            challenger = self._population[challenger_index]  # type: Genome
            challenger_fitness = challenger.get_fitness()
            if winner_fitness < challenger_fitness:
                losers.append(winner)
                winner = challenger

        for loser in losers:  # type: Genome
            winners_gene = winner.copy_gene()
            loser.set_gene(winners_gene)

        return


def run(gp, generation, size, step):
    for i in range(generation):
        print("# Generation: %d" % i)
        gp.init_world()
        sess = tf.Session()
        with sess.as_default():
            tf.global_variables_initializer().run()
            for _ in range(step):
                input = np.array([[[1 for _ in range(Genome.NUM_FEATURE)]] for _ in range(size)])
                output = gp.play(input)

            # set fitness
            gp.selection()
            gp.mutation()


def main(args):
    time_start = time.time()
    np.random.seed(0)

    generation = 30
    step = 500  # 各個体が何ステップ動くか
    size = 200  # Population size
    gp = GenePool(size)
    run(gp, generation, size, step)
    time_end = time.time()
    print("time: {}s".format(time_end - time_start))


if __name__ == '__main__':
    tf.app.run()



